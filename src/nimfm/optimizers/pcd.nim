import ../dataset, ../tensor/tensor, ../kernels, ../extmath
import ../models/factorization_machine, ../models/fm_base
import fit_linear, optimizer_base, utils
import math, sugar
from ../loss import newSquared
from ../regularizers/regularizers import newSquaredL12
from coordinate_descent import computeDerivative, update


type
  PCD*[L, R] = ref object of BaseCSCOptimizer
    ## Proximal coordinate descent solver.
    gamma*: float64
    loss*: L
    reg*: R


proc newPCD*[L, R](maxIter = 100, alpha0=1e-6, alpha=1e-3, beta=1e-4, 
                   gamma=1e-4, loss: L = newSquared(),
                   reg: R = newSquaredL12(), verbose = 1,
                   tol = 1e-3): PCD[L, R] =
  ## Creates new PCD.
  ## maxIter: Maximum number of iteration. At each iteration,
  ##          all parameters are updated once by using all samples.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## gamma: Sparsity-inducing-regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## reg: Sparsity-inducing regularization.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  result = PCD[L, R](maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, 
                     gamma: gamma, loss: loss, reg: reg, tol: tol,
                     verbose: verbose)


proc epoch[L, R](X: ColDataset, y: seq[float64], yPred: var Vector,
                 P: var Matrix, beta, gamma: float64, degree: int,
                 loss: L, A: var Matrix, dA: var Vector, reg: R): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[0]
  reg.computeCacheCDAll(P, degree)
  for s in 0..<nComponents:
    # compute cache
    anova(X, P, A, degree, s)
    reg.computeCacheCD(P, degree, s)
    for j in 0..<nFeatures:
      let psj = P[s, j]
      var (update, invStepSize) = update(psj, X, y, yPred, beta, degree, 
                                         j, loss, A, dA)
      if invStepSize < 1e-12: continue
      update /= invStepSize
      P[s, j] = reg.prox(psj, update, gamma/invStepSize, degree, s, j)

      update = psj - P[s, j]
      result += abs(update)
      # synchronize
      for (i, val) in X.getCol(j):
        dA[0] = val
        for deg in 1..<degree:
          dA[deg] = val * (A[i, deg] - psj * dA[deg-1])
          A[i, deg] -= update * dA[deg-1]
        A[i, degree] -= update * dA[degree-1]
        yPred[i] -= update * dA[degree-1]
      reg.updateCacheCD(P, degree, s, j)


# optimized proc for degree=2
proc epochDeg2[L, R](X: ColDataset, y: seq[float64], yPred: var Vector,
                     P: var Matrix, beta, gamma: float64, loss: L,
                     cacheDeg2: var Vector, reg: R): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[0]
  reg.computeCacheCDAll(P, 2) # for sparse regularization
  for s in 0..<nComponents:
    # compute cache. cacheDeg2[i] = \langle p_{s}, x_i \rangle
    cacheDeg2[0..^1] = 0
    for j in 0..<nFeatures:
      for (i, val) in X.getCol(j):
        cacheDeg2[i] += val * P[s, j]

    reg.computeCacheCD(P, 2, s) # for sparse regularization
    for j in 0..<nFeatures:
      let psj = P[s, j]
      var update = beta * psj
      var invStepSize = 0.0
      for (i, val) in X.getCol(j):
        let dA = (cacheDeg2[i] - psj * val) * val
        update += loss.dloss(y[i], yPred[i]) * dA
        invStepSize += dA^2
      invStepSize = invStepSize*loss.mu + beta
      if invStepSize < 1e-12: 
        continue
      update /= invStepSize
      P[s, j] = reg.prox(psj, update, gamma/invStepSize, 2, s, j) 
      update = psj - P[s, j]
      result += abs(update)
      # synchronize
      for (i, val) in X.getCol(j):
        yPred[i] -= update * (cacheDeg2[i] - psj * val) * val
        cacheDeg2[i] -= update * val
      reg.updateCacheCD(P, 2, s, j)


proc fit*[L, R](self: PCD[L, R], X: ColDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (PCD[L, R], FactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by coordinate descent.
  sfm.init(X)
  let y = sfm.checkTarget(y)
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    nComponents = sfm.P.shape[1]
    nOrders = sfm.P.shape[0]
    degree = sfm.degree
    alpha0 = self.alpha0 * float(nSamples)
    alpha = self.alpha * float(nSamples)
    beta = self.beta * float(nSamples)
    gamma = self.gamma * float(nSamples)
    fitLinear = sfm.fitLinear
    fitIntercept = sfm.fitIntercept
    nAugments = sfm.nAugments

  # caches
  var
    yPred = zeros([nSamples])
    A: Matrix = zeros([nSamples, degree+1])
    dA: Vector = zeros([degree])
    cacheDeg2: Vector = zeros([nSamples])
    colNormSq: Vector
    isConverged = false
  
  # init caches
  A[0..^1, 0] = 1.0
  if fitLinear:
    colNormSq = norm(X, axis=0, p=2)
    colNormSq *= colNormSq

  # compute prediction
  linear(X, sfm.w, yPred)
  yPred += sfm.intercept
  X.addDummyFeature(1.0, nAugments)
  for order in 0..<nOrders:
    for s in 0..<nComponents:
      anova(X, sfm.P[order], A, degree-order, s)
      yPred += A[0..^1, degree-order]

  self.reg.initCD(degree, nFeatures+nAugments, nComponents)
  if self.verbose > 0: # echo header
    echoHeader(self.maxIter)

  for it in 0..<self.maxIter:
    var viol = 0.0

    X.removeDummyFeature(nAugments)
    if fitIntercept:
      viol += fitInterceptCD(sfm.intercept, y, yPred, nSamples, alpha0, 
                             self.loss)

    if fitLinear:
      viol += fitLinearCD(sfm.w, X, y, yPred, colNormSq, alpha, self.loss)
    
    X.addDummyFeature(1.0, nAugments)
    for order in 0..<nOrders:
      if (degree-order) > 2:
        viol += epoch(X, y, yPred, sfm.P[order], beta, gamma, degree-order, 
                      self.loss, A, dA, self.reg)
      else:
        viol += epochDeg2(X, y, yPred, sfm.P[order], beta, gamma,
                          self.loss, cacheDeg2, self.reg)

    var lossVal = 0.0
    var regVal = 0.0
    if self.verbose > 0:
      for i in 0..<nSamples:
        lossVal += self.loss.loss(y[i], yPred[i])
      lossVal /= float(nSamples)
      for order in 0..<nOrders:
        regVal += gamma * self.reg.eval(sfm.P[order].T, degree-order)
      regVal += regularization(sfm.P, sfm.w, sfm.intercept,
                               alpha0, alpha, beta)
      regVal /= float(nSamples)

    if self.verbose > 0:
      echoInfo(it+1, self.maxIter, viol, lossVal, regVal)
      
    if not callback.isNil:
      callback(self, sfm)

    if viol < self.tol:
      if self.verbose > 0: echo("Converged at iteration ", it+1, ".")
      isConverged = true
      break

  X.removeDummyFeature(nAugments)
  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")