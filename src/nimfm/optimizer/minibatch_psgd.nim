import ../dataset, ../tensor/tensor, ../loss
import ../model/factorization_machine, ../model/fm_base, ../model/params
import optimizer_base, utils
from ../regularizer/regularizers import newSquaredL12
from sgd import predictWithGrad, getEta, SchedulingKind
from pgd import finalize
import math, sugar, sequtils, random, algorithm


type
  MBPSGD*[L, R] = ref object of BaseCSROptimizer
    gamma*: float64
    eta0: float64
    loss*: L
    reg*: R
    scheduling: SchedulingKind
    power: float64
    miniBatchSize: int
    maxIterInner: int
    shuffle: bool
    it: int
    nCalls: int


proc newMBPSGD*[L, R](maxIter=100, eta0=0.1, alpha0=1e-6, alpha=1e-3,
                      beta=1e-4, gamma=1e-4, loss: L=newSquared(),
                      reg: R=newSquaredL12(), miniBatchSize = -1,
                      maxIterInner = -1, scheduling = optimal,
                      power = 1.0, verbose = 1, tol = 1e-6,
                      shuffle=true, nCalls = -1): MBPSGD[L, R] =
  ## Creates new mini-batch proximal stochastic variance reduction
  ## gradient (MBPSGD).
  ## maxIter: Maximum number of iteration. At each iteration, all parameters
  ##          are updated maxIterInner times.
  ## eta0: step-size.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## gamma: Sparsity-inducing-regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## reg: Sparsity-inducing regularization.
  ## miniBatchSize: Number of samples in one minibatch. If <= 0, it is set to be 
  ##                minibatchsize = nFeatures * nSamples / nnz(X).
  ## maxIterInner: Maximum number of inner iteration (= the number of used
  ##               minibatches). If <= 0, it is set to be
  ##               maxIterInner = nSamples / miniBatchSize.
  ## scheduling: How to change the step-size.
  ##  - constant: eta = eta0,
  ##  - optimal: eta = eta0 / pow(1+eta0*regul*it, power),
  ##  - invscaling: eta = eta0 / pow(it, power),
  ##  - pegasos: eta = 1.0 / (regul * it),
  ##  where regul is the Regularization-strength hyperparameter.
  ## power: Hyperparameter for step size scheduling.
  ## shuffle: How to choose one instance: cyclic (false) or random permutation
  ##          (true).
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  
  result = MBPSGD[L, R](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, gamma: gamma,
    loss: loss, reg: reg, eta0: eta0, tol: tol, verbose: verbose, it: 0,
    scheduling: scheduling, power: power, miniBatchSize: miniBatchSize,
    maxIterInner: maxIterInner, shuffle: shuffle)


proc updateGradient*[L](X: RowDataset, y: seq[float64], i: int,
                        params, grads: Params, A: var Matrix, dA: var Tensor,
                        loss: L, degree, nAugments, miniBatchSize: int,
                        sign: float64): float64  =
  let yPred  = predictWithGrad(X, i, params.P, params.w, params.intercept, 
                               A, dA, degree, nAugments)
  result = loss.loss(y[i], yPred)
  let coef = sign * loss.dloss(y[i], yPred) / float(miniBatchSize)

  X.addDummyFeature(1.0, nAugments)
  for order in 0..<params.P.shape[0]:
    for j in X.getRowIndices(i):
      for s in 0..<params.P.shape[2]:
        grads.P[order, j, s] += coef * dA[order, j, s]
  X.removeDummyFeature(nAugments)

  if params.fitLinear:
    for (j, val) in X.getRow(i):
      grads.w[j] += coef * val
  
  if params.fitIntercept:
    grads.intercept += coef


proc epoch[L, R](self: MBPSGD[L, R], X: RowDataset, y: seq[float64],
                 params, grads: Params, A: var Matrix, dA: var Tensor,
                 indices: var seq[int], degree, nAugments: int,
                 miniBatchSize, maxIterInner: int, ii: var int): float64 =
  result = 0.0
  for itInner in 0..<maxIterInner:
    # compute gradient among a minibatch
    var b: int = 0
    grads <- 0.0

    # sample a minibatch
    while b < miniBatchSize:
      let i = indices[ii]
      result += updateGradient(X, y, i, params, grads, A, dA, self.loss,
                               degree, nAugments, miniBatchSize, 1.0)
      inc(b)
      inc(ii)
      if ii >= X.nSamples:
        ii = 0
        if X.nCached == X.nSamples and self.shuffle:
          shuffle(indices)
    
    # gradient update
    let eta_P = self.getEta(self.beta)
    let eta_w = self.getEta(self.alpha)
    let eta_intercept = self.getEta(self.alpha0)
    params.step(grads, eta_intercept, eta_w, eta_P, self.alpha0,
                self.alpha, self.beta)
    for order in 0..<params.P.shape[0]:
      self.reg.prox(params.P[order], self.gamma * eta_P / (1.0 + eta_P*self.beta),
                    degree-order)
    inc(self.it)

  result /= float(miniBatchSize * maxIterInner)


proc fit*[L, R](self: MBPSGD[L, R], X: RowDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (MBPSGD[L, R], FactorizationMachine)->void = nil) =
  ## Fits the sparse factorization machine on X and y by accelerated pgd.
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
    P: Tensor = zeros([nOrders, sfm.P.shape[2], nComponents])
    params = newParams(P, sfm.w, sfm.intercept, fitLinear, fitIntercept)
    A: Matrix = zeros([nComponents, degree+1])
    dA: Tensor  = zeros(P.shape)
    grads = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
    indices = toSeq(0..<nSamples)
    isConverged = false

  if not sfm.warmstart:
    self.it = 1

  # copy for fast training
  for order in 0..<sfm.P.shape[0]:
    P[order] = sfm.P[order].T

  var miniBatchSize = self.miniBatchSize
  if miniBatchSize <= 0:
    miniBatchSize = (nFeatures * nSamples) div X.nnz
    miniBatchSize = max(miniBatchSize, 1)
  var maxIterInner = self.maxIterInner
  if maxIterInner <= 0:
    maxIterInner = (nSamples-1) div miniBatchSize + 1
    maxIterInner = max(maxIterinner, 1)

  # perform optimization
  var ii = 0
  if X.nCached == X.nSamples and self.shuffle:
    shuffle(indices)
    
  self.reg.initSGD(degree, nFeatures+nAugments, nComponents)
  
  if self.verbose > 0: # echo header
    echo("Minibatch size: ", miniBatchSize)
    echo("Number of inner iteration: ", maxIterInner)
    echoHeader(self.maxIter, viol=false)
  
  var oldLossVal = Inf
  for it in 0..<self.maxIter:
    # perform inner loop
    let runningLoss = epoch(self, X, y, params, grads, A, dA, indices, degree,
                            nAugments, miniBatchSize, maxIterInner, ii)
    
    if not callback.isNil:
      finalize(sfm, params)
      callback(self, sfm)  
    
    if runningLoss.classify == fcNan:
      echo("Loss is NaN. Use smaller learning rate.")
      break

    if self.verbose > 0:
      var regVal = regularization(P, params.w, params.intercept, self.alpha0,
                                  self.alpha, self.beta)
      for order in 0..<nOrders:
        regVal += self.gamma * self.reg.eval(P[order], sfm.degree-order)
      echoInfo(it+1, self.maxIter, -1, runningLoss, regVal)
    
    if abs(oldLossVal - runningLoss) < self.tol:
      if self.verbose > 0: echo("Converged at epoch ", it+1, ".")
      isConverged = true
      break

    oldLossVal = runningLoss

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(sfm, params)