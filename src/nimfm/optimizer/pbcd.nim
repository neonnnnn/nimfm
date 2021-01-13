import ../dataset, ../tensor/tensor, ../kernels, ../model/factorization_machine
import ../model/fm_base, ../extmath, ../loss
from ../regularizer/regularizers import newSquaredL21
import fit_linear, optimizer_base, utils
import sequtils, math, random, sugar

type
  PBCD*[L, R] = ref object of BaseCSCOptimizer
    ## Proximal block coordinate descent solver.
    gamma*: float64
    loss*: L
    reg*: R
    sigma: float64
    rho: float64
    maxSearch: int
    shrink: bool
    shuffle: bool


proc newPBCD*[L, R](maxIter = 100, alpha0=1e-6, alpha=1e-3, beta=1e-4, 
                    gamma=1e-4, loss: L=newSquared(), reg: R=newSquaredL21(),
                    verbose = 1, tol = 1e-3, sigma=0.01, rho=0.5, maxSearch=0,
                    shrink=false, shuffle=false): PBCD[L, R] =
  ## Creates new PBCD.
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
  ## rho: Paraneter for line search.
  ## sigma: Parameter for line search.
  ## maxSearch: Maximum number of iteration in line search for the step-size.
  ##            If 0, algorithm uses the Lipschitz constant of the gradient.
  ## shrink: Whether to shrink or not. If true, algorithm do not perform line search
  ##         when row vector is updated as zero vector.
  ## shuffle: Whether cyclic (false) or random permutation (true) order.
  result = PBCD[L, R](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, gamma: gamma,
    loss: loss, reg: reg, tol: tol, verbose: verbose, sigma: sigma, rho:rho,
    maxSearch: maxSearch, shrink: shrink, shuffle:shuffle)


proc computeDerivative*(X: ColDataset, pj: Vector, A, dA: var Tensor,
                        j, degree: int) {.inline.} =
  let nComponents = dA.shape[2]
  for (i, val) in X.getCol(j):
    for s in 0..<nComponents:
      dA[0, i, s] = val
  for deg in 1..<degree:
    for (i, val) in X.getCol(j):
      for s in 0..<nComponents:
        dA[deg, i, s] = val * (A[deg, i, s] - pj[s] * dA[deg-1, i, s])
  

proc precomputeAnova(X: ColDataset, P: Matrix, A: var Tensor, degree: int) =
  ## Computes ANOVA kernels between all samples and all base vectors.
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    nComponents = P.shape[1]
  
  for t in 1..<degree+1:
    for i in 0..<nSamples:
      for s in 0..<nComponents:
        A[t, i, s] = 0

  for j in 0..<nFeatures:
    for t in 0..<degree:
      for (i, val) in X.getCol(j):
        for s in 0..<nComponents:
          A[degree-t, i, s] += A[degree-t-1, i, s] * val * P[j, s]


proc lineSearch[L, R](self: PBCD[L, R], X: ColDataset, y: seq[float64], 
                      yPred: var seq[float64], P: var Matrix, degree, j: int,
                      A, dA: var Tensor, delta: var Vector, cond:float64,
                      newLossVal: var float64, oldLossVal,
                      oldRegVal: float64): float64 =
  let nComponents = P.shape[1]
  let nSamples = float(X.nSamples)
  var alpha = 1.0
  var newRegVal = self.gamma*self.reg.value + 0.5*self.beta*norm(P[j], 2)^2
  var decreasing = (newLossVal / nSamples) + newRegVal
  decreasing -= (oldLossVal / nSamples) + oldRegVal

  var it = 0
  while not (decreasing <= self.sigma * alpha * cond):
    if it >= self.maxSearch: break
    for s in 0..<nComponents:
      P[j, s] -= alpha * (self.rho - 1) * delta[s]
    for (i, val) in X.getCol(j):
      newLossVal -= self.loss.loss(y[i], yPred[i])
      yPred[i] -= dot(dA[degree-1, i], delta) * alpha * (self.rho-1)
      newLossVal += self.loss.loss(y[i], yPred[i])

    alpha *= self.rho
    self.reg.updateCacheBCD(P, degree, j)
    newRegVal = 0.5*self.beta*norm(P[j], 2)^2 + self.gamma*self.reg.value
    decreasing = (newLossVal / nSamples) + newRegVal
    decreasing -= (oldLossVal / nSamples) + oldRegVal
    inc(it)
  delta *= alpha
  result = norm(delta, 1)


proc update[L, R](self: PBCD[L, R], P: var Matrix, X: ColDataset,
                  y: seq[float64], yPred: seq[float64], degree, j: int,
                  A, dA: var Tensor, grad, invStepSizes, delta: var Vector) =
  let
    nComponents = P.shape[1]
    nSamples = X.nSamples

  for s in 0..<nComponents:
    grad[s] = 0
    invStepSizes[s] = 0

  # computes grad and invStepSizes for all s \in [nComponents]
  if degree > 2: computeDerivative(X, P[j], A, dA, j, degree)
  else:
    for (i, val) in X.getCol(j):
      for s in 0..<nComponents:
        dA[1, i, s] = (val * (A[1, i, s] - val * P[j, s]))

  for (i, val) in X.getCol(j):
    let dL =  self.loss.dloss(y[i], yPred[i])
    for s in 0..<nComponents:
      grad[s] += dL * dA[degree-1, i, s]
      invStepSizes[s] += dA[degree-1, i, s]^2

  grad /= float(nSamples)
  for s in 0..<nComponents:
    grad[s] += self.beta * P[j, s]
  var invStepSize: float64
  # line search
  if self.maxSearch != 0:
    invStepSizes *= self.loss.mu / float(nSamples)
    invStepSize = max(invStepSizes)
  else: # without line search
    invStepSize = sum(invStepSizes) * self.loss.mu / float(nSamples)

  invStepSize += self.beta
  invStepSize = max(invStepSize, 1e-12)
  # updates
  for s in 0..<nComponents:
    delta[s] = P[j, s] # old value
    P[j, s] -= grad[s] / invStepSize
  
  self.reg.prox(P[j], self.gamma/invStepSize, degree, j)
  self.reg.updateCacheBCD(P, degree, j)
  for s in 0..<nComponents:
    delta[s] = - P[j, s] + delta[s] # delta = -new + old


proc epoch[L, R](self: PBCD[L, R], X: ColDataset, y: seq[float64],
                 yPred: var seq[float64], P: var Matrix,
                 degree: int, A, dA: var Tensor,
                 grad, invStepSizes, delta: var Vector,
                 lossVal: var float64): float64 =
  result = 0.0
  let 
    nFeatures = X.nFeatures
    nComponents = A.shape[2]
  var
    newLossVal, oldLossVal, oldRegVal, cond: float64
    indices = toSeq(0..<nFeatures)
  if self.shuffle: shuffle(indices)

  oldLossVal = lossVal
  self.reg.computeCacheBCD(P, degree)
  for j in indices:
    newLossVal = oldLossVal
    oldRegVal = 0.5*self.beta*norm(P[j], 2)^2 + self.gamma*self.reg.value
    cond = - self.gamma * self.reg.value

    update(self, P, X, y, yPred, degree, j, A, dA, grad, invStepSizes, delta)

    # synchronize yPred
    for (i, val) in X.getCol(j):
      newLossVal -= self.loss.loss(y[i], yPred[i])
      yPred[i] -= dot(delta, dA[degree-1, i])
      newLossVal += self.loss.loss(y[i], yPred[i])

    # without line search
    if self.maxSearch == 0:
      result += norm(delta, 1)
    else: # perform line search
      cond += self.gamma * self.reg.value
      cond -= dot(grad, delta)
      result += lineSearch(self, X, y, yPred, P, degree, j,  A, dA, delta,
                           cond, newLossVal, oldLossVal, oldRegVal)
    # synchronize A
    if degree == 2:
      for (i, val) in X.getCol(j):
        for s in 0..<nComponents:
          A[1, i, s] -= val * delta[s]
    else:
      for deg in 1..<degree:
        for (i, val) in X.getCol(j):
          for s in 0..<nComponents:
            A[deg, i, s] -= dA[deg-1, i, s] * delta[s]
    oldLossVal = newLossVal

  lossVal = newLossVal


proc fit*[L, R](self: PBCD[L, R], X: ColDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (self: PBCD[L, R], FactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by coordinate descent.
  ## Update pj \in R^d at each iteration.
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
    fitLinear = sfm.fitLinear
    fitIntercept = sfm.fitIntercept
    nAugments = sfm.nAugments

  # caches
  var
    yPred = newSeqWith(nSamples, 0.0)
    A: Tensor = zeros([degree+1, nSamples, nComponents])
    dA: Tensor = zeros([degree, nSamples, nComponents])
    grad: Vector = zeros([nComponents])
    delta: Vector = zeros([nComponents])
    invStepSizes: Vector = zeros([nComponents])
    colNormSq: Vector
    isConverged = false
    lossVal, regVal: float64
    P: Tensor = zeros([nOrders, nFeatures+nAugments, nComponents])
  
  # for fast training
  for order in 0..<nOrders:
    for j in 0..<nFeatures+nAugments:
      for s in 0..<nComponents:
        P[order, j, s] = sfm.P[order, s, j]
  
  # init caches
  for i in 0..<nSamples:
    for s in 0..<nComponents:
      A[0, i, s] = 1.0

  if fitLinear:
    colNormSq = norm(X, axis=0, p=2)
    colNormSq *= colNormSq

  # compute prediction
  linear(X, sfm.w, yPred)
  yPred += sfm.intercept

  X.addDummyFeature(1.0, nAugments)
  for order in 0..<nOrders:
    precomputeAnova(X, P[order], A, degree-order)
    for i in 0..<nSamples:
      for s in 0..<nComponents:
        yPred[i] += A[degree-order, i, s]
  
  self.reg.initBCD(degree, nFeatures+nAugments, nComponents)
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
    
    # compute loss value
    lossVal = 0.0
    for i in 0..<nSamples:
      lossVal += self.loss.loss(y[i], yPred[i])
    
    X.addDummyFeature(1.0, nAugments)
    if nOrders == 1:
      viol += epoch(self, X, y, yPred, P[0], degree, A, dA, grad,
                    invStepSizes, delta, lossVal)
    else:
      for order in 0..<nOrders:
        precomputeAnova(X, P[order], A, degree-order)
        viol += epoch(self, X, y, yPred, P[order], degree-order, A, dA, grad,
                      invStepSizes, delta, lossVal)

    lossVal /= float(nSamples)
    if self.verbose > 0:
      regVal = regularization(P, sfm.w, sfm.intercept, self.alpha0, self.alpha,
                              self.beta)
      for order in 0..<nOrders:
        regVal += self.gamma * self.reg.eval(P[order], degree-order)
      echoInfo(it+1, self.maxIter, viol, lossVal, regVal)

    if not callback.isNil:
      for order in 0..<nOrders:
        for j in 0..<nFeatures+nAugments:
          for s in 0..<nComponents:
            sfm.P[order, s, j] = P[order, j, s]
      callback(self, sfm)

    if viol < self.tol:
      if self.verbose > 0: 
        echo("Converged at iteration ", it+1, ".")
      isConverged = true
      break

  X.removeDummyFeature(nAugments)
  # finalize
  for order in 0..<nOrders:
    for j in 0..<nFeatures+nAugments:
      for s in 0..<nComponents:
        sfm.P[order, s, j] = P[order, j, s]
  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")