import ../dataset, ../tensor/tensor, ../extmath
import ../models/factorization_machine, ../models/fm_base, ../models/params
import optimizer_base, utils
from ../loss import newSquared
from ../regularizers/regularizers import newSquaredL12
from sgd import computeAnova, computeAnovaDerivative
import math, sugar


type
  PGD*[L, R] = ref object of BaseCSROptimizer
    gamma*: float64
    loss*: L
    reg*: R
    rho: float64
    sigma: float64
    maxSearch: int


proc newPGD*[L, R](maxIter=100, alpha0=1e-6, alpha=1e-3, beta=1e-4,
                   gamma=1e-4, loss: L=newSquared(),
                   reg: R=newSquaredL12(), rho=0.5, sigma=1.0, maxSearch = -1,
                   verbose = 1, tol = 1e-6): PGD[L, R] =
  ## Creates new PGD.
  ## maxIter: Maximum number of iteration. At each iteration,
  ##          all parameters are updated once by using all samples.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## gamma: Sparsity-inducing-regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## reg: Sparsity-inducing regularization.
  ## rho: Paraneter for line search. (0, 1)
  ## sigma: Parameter for line search. (0, 1]
  ## maxSearch: Maximum number of iterations in line search. If <= 0,
  ##            line search runs until the stopping condition is satisfied.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  result = PGD[L, R](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, gamma: gamma,
    loss: loss, reg: reg, rho: rho, sigma: sigma, maxSearch: maxSearch,
    tol: tol, verbose: verbose)


proc finalize*(sfm: FactorizationMachine, params: Params) =
  for order in 0..<params.P.shape[0]:
    for j in 0..<params.P.shape[1]:
      for s in 0..<params.P.shape[2]:
        sfm.P[order, s, j] = params.P[order, j, s]
  sfm.w = params.w
  sfm.intercept = params.intercept



proc predictAll*(X: RowDataset, yPred: var Vector, params: Params,
                 A: var Matrix, degree, nAugments: int) =
  let nSamples = X.nSamples
  yPred[0..^1] = 0.0
  # linear term
  mvmul(X, params.w, yPred)
  yPred += params.intercept

  # interaction term and its gradient
  X.addDummyFeature(1.0, nAugments)
  for i in 0..<nSamples:
    for order in 0..<params.P.shape[0]:
      yPred[i] += computeAnova(params.P[order], X, i, degree-order, A)
  X.removeDummyFeature(nAugments)


proc predictAllWithGrad*[L](X: RowDataset, y: seq[float64], yPred: var Vector,
                            params, grads: Params, A: var Matrix,
                            dA: var Tensor, dL: var Vector, loss: L,
                            degree, nAugments: int) =
  let nSamples = X.nSamples
  yPred[0..^1] = 0.0
  grads.P <- 0.0

  # linear term
  mvmul(X, params.w, yPred)
  yPred += params.intercept

  # interaction term and its gradient
  X.addDummyFeature(1.0, nAugments)
  for i in 0..<nSamples:
    for order in 0..<params.P.shape[0]:
      yPred[i] += computeAnova(params.P[order], X, i, degree-order, A)
      computeAnovaDerivative(params.P[order], X, i, degree-order, A, dA[order])
    # compute gradient
    dL[i] = loss.dloss(y[i], yPred[i])
    for order in 0..<params.P.shape[0]:
      for j in X.getRowIndices(i):
        for s in 0..<params.P.shape[2]:
          grads.P[order, j, s] += dL[i] * dA[order, j, s]
  X.removeDummyFeature(nAugments)

  # linear term gradient
  if params.fitLinear:
    vmmul(dL, X, grads.w)
  if params.fitIntercept:
    grads.intercept = sum(dL)
    
  # scale
  grads /= float(nSamples)


proc linesearch*[L, R](X: RowDataset, y: seq[float64], yPred: var Vector,
                       params, old_params, grads: Params, A: var Matrix,
                       alpha0, alpha, beta, gamma: float64,
                       loss: L, reg: R, degree, nAugments: int,
                       etaInit, rho, sigma: float64,
                       maxSearch: int): (float64, float64) =
  var eta = etaInit
  var it: int = 0
  var oldLoss = 0.0
  for i in 0..<X.nSamples:
    oldLoss += loss.loss(y[i], yPred[i])
  oldLoss /= float(X.nSamples)
  
  while it < maxSearch or maxSearch <= 0:
    # graident step
    params <- old_params
    params.step(grads, eta, eta, eta, alpha0, alpha, beta)
    # proximal step
    for order in 0..<params.P.shape[0]:
      reg.prox(params.P[order], gamma * eta / (1.0 + eta*beta), degree-order)

    # predict and compute loss
    predictAll(X, yPred, params, A, degree, nAugments)
    result[0] = 0.0
    for i in 0..<X.nSamples:
      result[0] += loss.loss(y[i], yPred[i])
    result[0] /= float(X.nSamples)

    # compute quadratic approximation
    var cond = dot(params, grads) - dot(old_params, grads)
    cond += 0.5 * computeViol(params, old_params) / eta
    # stop?
    if (result[0] - oldLoss) <= sigma * cond or eta < 1e-12:
      break
    eta *= rho
    inc(it)

  # compute regularization term
  result[1] = regularization(params, alpha0, alpha, beta)
  for order in 0..<params.P.shape[0]:
    result[1] += gamma * reg.eval(params.P[0], degree-order)

    
proc fit*[L, R](self: PGD[L, R], X: RowDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (PGD[L, R], FactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  sfm.init(X)
  let y = sfm.checkTarget(y)
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    nComponents = sfm.P.shape[1]
    nOrders = sfm.P.shape[0]
    degree = sfm.degree
    fitLinear = sfm.fitLinear
    fitIntercept = sfm.fitIntercept
    nAugments = sfm.nAugments
  var
    yPred: Vector = zeros([nSamples])
    dL: Vector = zeros([nSamples])
    P: Tensor = zeros([nOrders, sfm.P.shape[2], nComponents])
    params: Params
    old_params = newParams(P.shape, nFeatures, fitLinear, fitIntercept)
    grads = newParams(P.shape, nFeatures, fitLinear, fitIntercept)
    A: Matrix = zeros([nComponents, degree+1])
    dA: Tensor = zeros(P.shape)
    isConverged = false
  
  # copy for fast training
  for order in 0..<nOrders:
    P[order] = sfm.P[order].T
  params = newParams(P, sfm.w, sfm.intercept, fitLinear, fitIntercept)

  self.reg.initSGD(degree, nFeatures+nAugments, nComponents)
  if self.verbose > 0: # echo header
    echoHeader(self.maxIter, viol=true)

  # perform optimization
  var viol = Inf
  for epoch in 0..<self.maxIter:
    # update parameters
    old_params <- params
    predictAllWithGrad(X, y, yPred, params, grads, A, dA, dL, self.loss,
                       degree, nAugments)
    var (lossVal, regVal) = lineSearch(
      X, y, yPred, params, old_params, grads, A, self.alpha0, self.alpha,
      self.beta, self.gamma, self.loss, self.reg, degree, nAugments, 1.0,
      self.rho, self.sigma, self.maxSearch)

    # callback, stoping criterion, and echo information
    if not callback.isNil:
      finalize(sfm, params)
      callback(self, sfm)  
    
    viol = Inf
    if self.tol > 0 or self.verbose > 0:
      viol = computeViol(params, old_params)

    if self.verbose > 0:
      echoInfo(epoch+1, self.maxIter, viol, lossVal, regVal)
    
    if viol < self.tol:
      if self.verbose > 0: echo("Converged at epoch ", epoch, ".")
      isConverged = true
      break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(sfm, params)