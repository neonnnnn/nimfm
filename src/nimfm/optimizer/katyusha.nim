import ../dataset, ../tensor/tensor, ../loss
import ../model/factorization_machine, ../model/fm_base, ../model/params
import optimizer_base, utils
from ../regularizer/regularizers import newSquaredL12
from nmapgd import extrapolate
from minibatch_psgd import updateGradient
from pgd import predictAllWithGrad
import math, sugar, sequtils, random, algorithm


type
  Katyusha*[L, R] = ref object of BaseCSROptimizer
    gamma*: float64
    eta: float64
    loss*: L
    reg*: R
    miniBatchSize: int
    shuffle: bool
    tau1: float64
    tau2: float64
    nCalls: int


proc newKatyusha*[L, R](maxIter=100, eta=0.1, alpha0=1e-6, alpha=1e-3, 
                        beta=1e-4, gamma=1e-4, loss: L=newSquared(),
                        reg: R=newSquaredL12(), miniBatchSize = -1,
                        tau1 = 0.5, tau2 = -1.0, verbose = 1, tol = 1e-6,
                        shuffle=true, nCalls = -1): Katyusha[L, R] =
  ## Creates new Katyusha.
  ## maxIter: Maximum number of iteration. At each iteration, all parameters
  ##          are updated maxIterInner times.
  ## eta: step-size.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## gamma: Sparsity-inducing-regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## reg: Sparsity-inducing regularization.
  ## miniBatchSize: Number of samples in one minibatch. If <= 0, it is set to 
  ##                be minibatchsize = nFeatures * nSamples / nnz(X).
  ## tau1: momentum-hyperparameter. If < 0, tau1 = tau2.
  ## tau2: momentum-hyperparameter. If < 0, tau2 = 1 / (2.0 * miniBatchSize).
  ## shuffle: How to choose one instance: cyclic (false) or random permutation
  ##          (true).
  ## nCalls: Frequency with which callback must be called in the inner loop.
  ##         If nCalls <= 0, callback is called per one epoch.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  result = Katyusha[L, R](
    maxIter: maxIter, eta: eta, alpha0: alpha0, alpha: alpha, beta: beta, gamma: gamma,
    loss: loss, reg: reg,shuffle: shuffle, miniBatchSize: miniBatchSize, 
    tau1: tau1, tau2: tau2, tol: tol, verbose: verbose, nCalls: nCalls)


proc finalize(sfm: FactorizationMachine, tilde_params, y_params: Params,
              tau1, tau2, m: float64) =
  
  for order in 0..<sfm.P.shape[0]:
    for s in 0..<sfm.P.shape[1]:
      for j in 0..<sfm.P.shape[2]:
        sfm.P[order, s, j] = m * tau2 * tilde_params.P[order, j, s]
        sfm.P[order, s, j] += (1 - tau1 - tau2) * y_params.P[order, j, s]
  sfm.P /= tau2 * m + 1.0 - tau1 - tau2
  if sfm.fitLinear:
    for j in 0..<len(sfm.w):
      sfm.w[j] = m * tau2 * tilde_params.w[j]
      sfm.w[j] += (1 - tau1 - tau2) * y_params.w[j]
    sfm.w /= tau2 * m + 1.0 - tau1 - tau2
  if sfm.fitIntercept:
    sfm.intercept = m * tau2 * tilde_params.intercept
    sfm.intercept += (1 - tau1 - tau2) * y_params.intercept
    sfm.intercept /= tau2 * m + 1.0 - tau1 - tau2

  
proc epoch[L, R](self: Katyusha[L, R], X: RowDataset, y: seq[float64],
                 params, z_params, y_params, tilde_params: Params,
                 next_tilde_params, grads, grads_ave: Params,
                 A: var Matrix, dA: var Tensor, indices: var seq[int],
                 degree, nAugments, miniBatchSize, maxIterInner: int,
                 ii: var int, tau1, tau2: float64, sfm: FactorizationMachine,
                 callback: (Katyusha[L, R], FactorizationMachine)->void = nil) =
  let 
    m = float(maxIterInner)
    theta_P = 1.0 + min(self.eta * self.beta, 1.0 / (4.0*m))
    theta_w = 1.0 + min(self.eta * self.alpha, 1.0 / (4.0*m))
    theta_intercept = 1.0 + min(self.eta * self.alpha0, 1.0 / (4.0*m))
  var 
    theta_pow_P = 1.0
    theta_pow_w = 1.0
    theta_pow_intercept = 1.0

  next_tilde_params.P <- 0
  if params.fitLinear:
    next_tilde_params.w <- 0.0
  if params.fitIntercept:
    next_tilde_params.intercept = 0.0

  for itInner in 0..<maxIterInner:
    # compute x_{k+1}
    extrapolate(params, z_params, tilde_params, y_params, 
                tau1, tau2, 1-tau1-tau2)
    # compute gradient among a minibatch
    var b = 0
    grads <- grads_ave
    # sample a minibatch
    while b < miniBatchSize:
      let i = indices[ii]
      discard updateGradient(X, y, i, params, grads, A, dA, self.loss, degree,
                             nAugments, miniBatchSize, 1)
      discard updateGradient(X, y, i, tilde_params, grads, A, dA, self.loss,
                             degree, nAugments, miniBatchSize, -1)
      inc(b)
      inc(ii)
      if ii mod X.nSamples == 0:
        ii = 0
        if X.nCached == X.nSamples and self.shuffle:
          shuffle(indices)
    # gradient update
    step(z_params, grads, self.eta, self.eta, self.eta,
         self.alpha0, self.alpha, self.beta)
    for order in 0..<params.P.shape[0]:
      self.reg.prox(z_params.P[order],
                    self.gamma * self.eta / (1.0 + self.beta * self.eta),
                    degree-order)
    
    # Option 2
    # y_{k+1} = x_{k+1} + tau1 (z_{k+1} - z_{k})
    # = tau2 \tilde{x}_{k} + (1 - tau1 - tau2) y_{k} + tau1 z_{k+1}.
    y_params *= 1.0 - tau1 - tau2
    y_params.add(tilde_params, tau2)
    y_params.add(z_params, tau1)
    next_tilde_params.add(y_params, theta_pow_intercept, theta_pow_w,
                          theta_pow_P)
    theta_pow_P *= theta_P
    theta_pow_w *= theta_w
    theta_pow_intercept *= theta_intercept

    # callback
    if self.nCalls > 0 and (itInner + 1) mod self.nCalls == 0:
      let coef_P = (1.0 - theta_P) / (1.0 - theta_pow_P)
      let coef_w = (1.0 - theta_w) / (1.0 - theta_pow_w)
      let coef_intercept = (1.0 - theta_intercept) / (1.0 - theta_pow_intercept)
      next_tilde_params.scale(coef_intercept, coef_w, coef_P)
      finalize(sfm, next_tilde_params, y_params, tau1, tau2, float64(itInner+1))
      callback(self, sfm)
      next_tilde_params.scale(1.0/coef_intercept, 1.0/coef_w, 1.0/coef_P)

  # finalize next_tilde_P
  let coef_P = (1.0 - theta_P) / (1.0 - theta_pow_P)
  let coef_w = (1.0 - theta_w) / (1.0 - theta_pow_w)
  let coef_intercept = (1.0 - theta_intercept) / (1.0 - theta_pow_intercept)
  next_tilde_params.scale(coef_intercept, coef_w, coef_P)


proc fit*[L, R](self: Katyusha[L, R], X: RowDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (Katyusha[L, R], FactorizationMachine)->void = nil) =
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
    yPred: Vector = zeros([nSamples])
    dL: Vector = zeros([nSamples])
    P: Tensor = zeros([nOrders, sfm.P.shape[2], nComponents])
    A: Matrix = zeros([nComponents, degree+1])
    dA: Tensor  = zeros(P.shape)
    params, y_params, z_params, tilde_params, next_tilde_params: Params
    grads, grads_ave: Params
    indices = toSeq(0..<nSamples)
    isConverged = false

  # copy for fast training
  for order in 0..<sfm.P.shape[0]:
    P[order] = sfm.P[order].T
  params = newParams(P, sfm.w, sfm.intercept, fitLinear, fitIntercept)

  # init caches
  y_params = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
  y_params <- params
  tilde_params = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
  tilde_params <- params
  z_params = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
  z_params <- params
  next_tilde_params = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
  grads = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
  grads_ave = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)

  var ii = 0
  if X.nCached == X.nSamples and self.shuffle:
    shuffle(indices)

  # compute miniBatchSize, maxIterInner, tau1, and tau2
  var miniBatchSize = self.miniBatchSize
  if miniBatchSize <= 0:
    miniBatchSize = (nFeatures * nSamples) div X.nnz
    miniBatchSize = max(miniBatchSize, 1)
  var maxIterInner = ((nSamples-1) div miniBatchSize + 1)
  var tau2 = self.tau2
  if tau2 < 0:
    tau2 = 1.0 / (2.0 * float(miniBatchSize))
  var tau1 = self.tau1
  if tau1 < 0:
    tau1 = tau2

  # compute average gradient
  predictAllWithGrad(X, y, yPred, params, grads_ave, A, dA, dL, self.loss,
                     degree, nAugments)
  
  self.reg.initSGD(degree, nFeatures+nAugments, nComponents)

  if self.verbose > 0: # echo header
    echo("Minibatch size: ", miniBatchSize)
    echo("Number of inner iteration: ", maxIterInner)
    echoHeader(self.maxIter, viol=true)

  # perform optimization
  for it in 0..<self.maxIter:
    # perform inner loop
    epoch(self, X, y, params, z_params, y_params, tilde_params,
          next_tilde_params, grads, grads_ave, A, dA, indices, degree,
          nAugments, miniBatchSize, maxIterInner, ii, tau1, tau2, sfm,
          callback)
    var viol = Inf
    if self.tol > 0 or self.verbose > 0:
      viol = computeViol(next_tilde_params, tilde_params)   
    tilde_params <- next_tilde_params
    if not callback.isNil:
      finalize(sfm, tilde_params, y_params, float(maxIterInner), tau1, tau2)
      callback(self, sfm)  
    
    var lossVal = 0.0
    for i in 0..<nSamples:
      lossVal += self.loss.loss(y[i], yPred[i])
    lossVal /= float(nSamples)
    if lossVal.classify == fcNan:
      echo("Loss is NaN. Use smaller learning rate.")
      break

    if self.verbose > 0:
      var regVal = regularization(tilde_params, self.alpha0, self.alpha, self.beta)
      for order in 0..<nOrders:
        regVal += self.gamma * self.reg.eval(tilde_params.P[order], sfm.degree-order)
      echoInfo(it+1, self.maxIter, viol, lossVal, regVal)
    
    if viol < self.tol:
      if self.verbose > 0: echo("Converged at epoch ", it+1, ".")
      isConverged = true
      break
    
    # update average gradients for next iteration
    predictAllWithGrad(X, y, yPred, tilde_params, grads_ave, A, dA, dL,
                       self.loss, degree, nAugments)


  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(sfm, tilde_params, y_params, float(maxIterInner), tau1, tau2)
