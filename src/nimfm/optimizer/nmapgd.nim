import ../dataset, ../tensor/tensor, ../loss
import  ../model/factorization_machine, ../model/fm_base, ../model/params
import optimizer_base, utils
import math, sugar
from ../regularizer/regularizers import newSquaredL12
from pgd import predictAll, predictAllWithGrad, finalize


type
  NMAPGD*[L, R] = ref object of BaseCSROptimizer
    gamma*: float64
    loss*: L
    reg*: R
    t, c, q, eta: float64
    rho, sigma: float64
    maxSearch: int
    old_y_params, z_params, old_x_params, old_y_grads: Params


proc newNMAPGD*[L, R](maxIter = 100, alpha0=1e-6, alpha=1e-3, beta=1e-4,
                      gamma=1e-4, loss: L=newSquared(), reg: R=newSquaredL12(),
                      rho=0.5, sigma=0.01, maxSearch = -1, eta=0.50, verbose = 1,
                      tol = 1e-5): NMAPGD[L, R] =
  ## Creates new nonmonotone APGD (NMAPGD).
  ## maxIter: Maximum number of iterations. At each iteration,
  ##          all parameters are updated once by using all samples.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## gamma: Sparsity-inducing-regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## reg: Sparsity-inducing regularization.
  ## rho: Paraneter for line search. (0, 1)
  ## sigma: Parameter for line search. (0, 1)
  ## eta: Parameter for controlling the degree of nonmonotonicity
  ##      (not step size!). [0, 1)
  ## maxSearch: Maximum number of iterations in line search. If <=0,
  ##            line search runs until the stopping condition is satisfied.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  result = NMAPGD[L, R](maxIter: maxIter, rho: rho, sigma: sigma,
                        maxSearch: maxSearch, eta: eta, tol: tol,
                        verbose: verbose, alpha0: alpha, alpha: alpha,
                        beta: beta, gamma: gamma, loss: loss, reg: reg,
                        q: 1, c: -1, t: 0)


proc initCaches(self: NMAPGD, params: Params, degree: int, warmStart: bool,
                y_params, y_grads, x_grads: var Params,
                s_params, r_params: var Params) =
  let
    fitLinear = params.fitLinear
    fitIntercept = params.fitIntercept
    shape_P = params.P.shape
    len_w = params.w.len
  
  if not warmStart:
    self.t = 0.0
    self.c = -1.0
    self.q = 1.0
  
  if self.t == 0.0 or self.z_params.P.shape != params.P.shape:
    self.z_params = newParams(shape_P, len_w, fitLinear, fitIntercept)
    self.old_y_params = newParams(shape_P, len_w, fitLinear, fitIntercept)
    self.old_x_params = newParams(shape_P,len_w, fitLinear, fitIntercept)
    self.old_y_grads = newParams(shape_P, len_w, fitLinear, fitIntercept)
    self.old_y_params <- params
    self.old_x_params <- params
    self.reg.initSGD(degree, params.P.shape[1], params.P.shape[2])

  y_params = newParams(shape_P, len_w, fitlinear, fitIntercept)
  y_grads = newParams(shape_P, len_w, fitlinear, fitIntercept)
  x_grads = newParams(shape_P, len_w, fitlinear, fitIntercept)
  s_params = newParams(shape_P, len_w, fitlinear, fitIntercept)
  r_params = newParams(shape_P, len_w, fitlinear, fitIntercept)


proc updateCaches(self: NMAPGD, y_params, y_grads: Params,
                  t, loss, reg: float64) =
  self.old_y_params <- y_params
  self.old_y_grads <- y_grads
  self.t = t
  self.c = self.eta * self.c * self.q + loss + reg
  self.q = self.eta * self.q + 1
  self.c /= self.q


proc getStepSize(params, old_params, grads, old_grads: Params,
                 s_params, r_params: Params): float64 =
  s_params <- params
  s_params -= old_params
  r_params <- grads
  r_params -= old_grads
  var ss = dot(s_params, s_params)
  var sr = dot(s_params, r_params)
  if ss == 0.0 or sr == 0.0:
    result = 1.0
  else: result = abs(ss/sr)


proc linesearch[L, R](X: RowDataset, y: seq[float64], yPred: var Vector,
                      params, old_params, grads: Params, A: var Matrix,
                      dL: Vector, alpha0, alpha, beta, gamma: float64,
                      loss: L, reg: R, degree, nAugments: int,
                      etaInit, rho, sigma: float64, maxSearch: int,
                      c: float64): (float64, float64) =
  var eta = etaInit
  var it: int = 0
  while it < maxSearch or maxSearch <= 0:
    # gradient step
    params <- old_params
    params.step(grads, eta, eta, eta, alpha0, alpha, beta)

    # proximal step
    for order in 0..<params.P.shape[0]:
      reg.prox(params.P[order], gamma*eta / (1.0 + eta*beta), degree-order)

    # predict and compute loss
    predictAll(X, yPred, params, A, degree, nAugments)
    result = objective(y, yPred, params, alpha0, alpha, beta, loss)
    for order in 0..<params.P.shape[0]:
      result[1] += gamma * reg.eval(params.P[order], degree-order)

    # stop?
    let cond = computeViol(params, old_params)
    if (result[0] + result[1] - c) <= -sigma * cond or eta < 1e-12:
      break
    eta *= rho
    inc(it)


proc extrapolate*(y_params, z_params, x_params, old_x_params: Params,
                  tau1, tau2, tau3: float64) {.inline.} =
  y_params <- z_params
  y_params *= tau1
  y_params.add(x_params, tau2)
  y_params.add(old_x_params, tau3)
  

proc epochZ[L, R](self: NMAPGD[L, R], X: RowDataset, y: seq[float64],
                  yPred: var Vector, y_params, y_grads: Params,
                  s_params, r_params: Params, A: var Matrix, dA: var Tensor,
                  dL: var Vector, degree, nAugments: int): (float64, float64) =
  predictAllWithGrad(X, y, yPred, y_params, y_grads, A, dA, dL, self.loss,
                     degree, nAugments)
  let stepsize = getStepSize(y_params, self.old_y_params, y_grads,
                             self.old_y_grads, s_params, r_params)
  let (y_loss, y_reg) = objective(y, yPred, y_params, self.alpha0,
                                  self.alpha, self.beta, self.loss)
  var c = y_loss + y_reg
  for order in 0..<y_params.P.shape[0]:
    c += self.gamma * self.reg.eval(y_params.P[order], degree-order)
  result = lineSearch(X, y, yPred, self.z_params, y_params, y_grads,
                      A, dL, self.alpha0, self.alpha, self.beta, self.gamma,
                      self.loss, self.reg, degree, nAugments, stepsize,
                      self.rho, self.sigma, self.maxSearch, max(c, self.c))


proc epochV[L, R](self: NMAPGD[L, R], X: RowDataset, y: seq[float64],
                  yPred: var Vector, v_params, x_grads: Params,
                  s_params, r_params: Params, A: var Matrix, dA: var Tensor,
                  dL: var Vector, degree, nAugments: int): (float64, float64) =
  predictAllWithGrad(X, y, yPred, v_params, x_grads, A, dA, dL, self.loss,
                     degree, nAugments)
  let stepsize = getStepSize(v_params, self.old_y_params, x_grads,
                             self.old_y_grads, s_params, r_params)
  result = lineSearch(X, y, yPred, v_params, self.old_x_params, x_grads,
                      A, dL, self.alpha0, self.alpha, self.beta, self.gamma,
                      self.loss, self.reg, degree, nAugments, stepsize,
                      self.rho, self.sigma, self.maxSearch, self.c)


proc fit*[L, R](self: NMAPGD[L, R], X: RowDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (NMAPGD[L, R], FactorizationMachine)->void = nil) =
  ## Fits the sparse factorization machine on X and y by nonmonotone APGD.
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
    P: Tensor = zeros([nOrders, nFeatures+nAugments, nComponents])
    params: Params
    y_params, y_grads, x_grads, s_params, r_params: Params
    A: Matrix = zeros([nComponents, degree+1])
    dA: Tensor = zeros(P.shape)
    isConverged = false
    lossVal, regVal: float64

  # copy for fast training
  for order in 0..<sfm.P.shape[0]:
    P[order] = sfm.P[order].T
  params = newParams(P, sfm.w, sfm.intercept, fitLinear, fitIntercept)

  # init caches
  self.initCaches(params, degree, sfm.warmStart, y_params, y_grads,
                  x_grads, s_params, r_params)
  # init c
  if self.c < 0:
    predictAll(X, yPred, params, A, degree, nAugments)
    (lossVal, regVal) = objective(y, yPred, params, self.alpha0,
                                  self.alpha, self.beta, self.loss)
    for order in 0..<P.shape[0]:
      regVal += self.gamma * self.reg.eval(P[order], degree-order)
    self.c = lossVal + regVal
    
  if self.verbose > 0:
    echoHeader(self.maxIter, viol=true)

  # perform optimization
  for it in 0..<self.maxIter:
    let t = (sqrt(4*self.t^2+1.0)+1.0) / 2.0
    extrapolate(y_params, self.z_params, params, self.old_x_params,
                self.t/t,  (t-1)/t, -(self.t-1)/t)

    self.old_x_params <- params
    let (z_loss, z_reg) = epochZ(self, X, y, yPred, y_params, y_grads,
                                 s_params, r_params, A, dA, dL, degree,
                                 nAugments)
    
    let cond = computeViol(self.z_params, y_params)
    var v_loss = Inf
    var v_reg = Inf
    if (z_loss + z_reg) > self.c - self.sigma * cond:
      (v_loss, v_reg) = epochV(self, X, y, yPred, params, x_grads,
                               s_params, r_params, A, dA, dL, degree,
                               nAugments)
    if (z_loss + z_reg) <= (v_loss + v_reg):
      lossVal = z_loss
      regVal = z_reg
      params <- self.z_params
    else:
      lossVal = v_loss
      regVal = v_loss

    # update caches for next iteration
    updateCaches(self, y_params, y_grads, t, lossVal, regVal)

    if not callback.isNil:
      finalize(sfm, params)
      callback(self, sfm)  

    var viol = Inf
    if self.tol > 0.0 or self.verbose > 0:
      viol = computeViol(params, self.old_x_params)     
    
    if self.verbose > 0:
      echoInfo(it+1, self.maxIter, viol, lossVal, regVal)

    if viol < self.tol:
      if self.verbose > 0: echo("Converged at epoch ", it+1, ".")
      isConverged = true
      break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  finalize(sfm, params)