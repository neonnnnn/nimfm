import ../dataset, ../tensor/tensor, ../loss
import ../model/factorization_machine, ../model/fm_base, ../model/params
import optimizer_base, utils
from ../regularizer/regularizers import newSquaredL12
from pgd import predictAll, predictAllWithGrad, linesearch, finalize
import math, sugar


type
  FISTA*[L, R] = ref object of BaseCSROptimizer
    gamma*: float64
    loss*: L
    reg*: R
    rho: float64
    sigma: float64
    maxSearch: int
    t: float64


proc newFISTA*[L, R](maxIter=100, alpha0=1e-6, alpha=1e-3, beta=1e-4,
                   gamma=1e-4, loss: L=newSquared(), reg: R=newSquaredL12(), 
                   rho=0.5, sigma=1.0, maxSearch = -1, verbose = 1,
                   tol = 1e-6): FISTA[L, R] =
  ## Creates new FISTA.
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
  result = FISTA[L, R](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, gamma: gamma,
    loss: loss, reg: reg, rho: rho, sigma: sigma, maxSearch: maxSearch,
    tol: tol, verbose: verbose, t: 0)


proc extrapolate*(z_params, params, old_params: Params, coef: float64) =
  z_params <- params
  z_params.add(params, coef)
  z_params.add(old_params, -coef)


proc fit*[L, R](self: FISTA[L, R], X: RowDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (FISTA[L, R], FactorizationMachine)->void = nil) =
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
    params: Params
    old_params = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
    z_params = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
    grads = newParams(P.shape, sfm.w.len, fitLinear, fitIntercept)
    A: Matrix = zeros([nComponents, degree+1])
    dA: Tensor = zeros(P.shape)
    isConverged = false 

  # copy for fast training
  for order in 0..<sfm.P.shape[0]:
    P[order] = sfm.P[order].T
  params = newParams(P, sfm.w, sfm.intercept, fitLinear, fitIntercept)

  # compute caches
  old_params <- params
  z_params <- params

  if not sfm.warmStart:
    self.t = 0
  
  self.reg.initSGD(degree, nFeatures+nAugments, nComponents)
  
  if self.verbose > 0: # echo header
    echoHeader(self.maxIter, viol=true)

  # perform optimization
  var lossVal = Inf
  var regVal = Inf
  for it in 0..<self.maxIter:
    let t = (sqrt(4*self.t^2+1.0)+1.0) / 2.0
    # compute z_{k+1}
    extrapolate(z_params, params, old_params, (self.t-1)/t)
    old_params <- z_params
    predictAllWithGrad(X, y, yPred, z_params, grads, A, dA, dL,
                       self.loss, degree, nAugments)
    var (z_loss, z_reg) = lineSearch(
      X, y, yPred, z_params, old_params, grads, A, self.alpha0, self.alpha,
      self.beta, self.gamma, self.loss, self.reg, degree, nAugments, 1.0,
      self.rho, self.sigma, self.maxSearch)
    
    # Accept? Accept!
    if (z_loss + z_reg) <= (lossVal + regVal):
      lossVal = z_loss
      regVal = z_reg
      old_params <- params
      params <- z_params
      self.t = t
    else: # Restart!
      self.t = 1.0
    
    if not callback.isNil:
      finalize(sfm, params)
      callback(self, sfm)  
    
    var viol = Inf
    if self.tol > 0 or self.verbose > 0:
      viol = computeViol(params, old_params)

    if self.verbose > 0:
      echoInfo(it+1, self.maxIter, viol, lossVal, regVal)
    
    if viol < self.tol:
      if self.verbose > 0: echo("Converged at epoch ", it+1, ".")
      isConverged = true
      break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(sfm, params)