import nimfm/tensor/tensor, nimfm/optimizer/optimizer_base, nimfm/utils
from nimfm/model/fm_base import checkTarget, checkInitialized
import math, ../model/cfm_slow, ../kernels_slow

type
  HazanSlow* = ref object of BaseCSCOptimizer
    ## Hazan's algorithm for convex factorization machines with SquaredLoss.
    ## In this solver, the regularization for P is not 
    ## squared Frobenius norm but the trace norm for interaction weight
    ## matrix. 
    ## This solver solves not regularized problem but constrained problem.
    ## Regularization parameters alpha0, alpha, and beta
    ## in ConvexFactorizationMachine are ignored.
    eta: float64
    maxIterPower: int
    tolPower: float64
    optimal: bool
    nTol: int
    it: int


proc newHazanSlow*(
  maxIter = 100, eta=1000.0, verbose = 2, tol = 1e-7, nTol=10, 
  maxIterPower = 1000, tolPower = 1e-7, optimal = true): HazanSlow =
  result = HazanSlow(
    maxIter: maxIter, eta: eta, tol: tol, nTol: nTol, verbose: verbose,
    maxIterPower: maxIterPower, tolPower: tolPower, optimal: optimal)


proc predict(yPredQuad, yPredLinear: var Vector, K: var Matrix,
             X, P: Matrix, lams, w: Vector, intercept: float,
             ignoreDiag: bool) = 
  let nSamples = X.shape[0]
  let nFeatures = X.shape[1]
  mvmul(X, w, yPredLinear)
  yPredLinear += intercept
  yPredQuad[0..^1] = 0.0
  for s in 0..<len(K):
    for i in 0..<nSamples:
      if ignoreDiag: 
        K[s, i] = anovaSlow(X, P, i, 2, s, nFeatures, 0)
      else: 
        K[s, i] = polySlow(X, P, i, 2, s, nFeatures, 0)
    yPredQuad += lams[s] * K[s]


proc fit*(self: HazanSlow, X: Matrix, y: seq[float64],
          cfm: var CFMSlow) =
  ## Fits the factorization machine on X and y by coordinate descent.
  cfm.init(X)
  let y = checkTarget(cfm, y)
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    fitLinear = cfm.fitLinear
    fitIntercept = cfm.fitIntercept
    ignoreDiag = cfm.ignoreDiag

  # caches
  var
    yPredLinear: Vector = zeros([nSamples])
    yPredQuad: Vector = zeros([nSamples])
    residual: Vector = zeros([nSamples])
    cacheK: Vector = zeros([nSamples])
    K: Matrix = zeros([len(cfm.lams), nSaMples])
    Z: Matrix = zeros(X.shape)
    ZTZ: Matrix
    w: Vector
    resZ: Vector
    colNormSq: Vector
    Preconditioner: Matrix
    lossOld = 0.0
    lossNew = 0.0
  
  for i in 0..<nSamples:
    Z[i] = X[i]
  if not cfm.warmStart:
    self.it = 0

  if fitLinear:
    w = cfm.w[0..<nFeatures]
    resZ = zeros([nFeatures])
    colNormSq = zeros([nFeatures])
    for i in 0..<nSamples:
      for j in 0..<nFeatures:
        colNormSq[j] += X[i, j]^2
    if fitIntercept: 
      Z.addCol(ones([nSamples]))
      w.add(cfm.intercept)
      colNormSq.add(float(nSamples))
      resZ.add(0.0)
    colNormSq += 1e-5

    Preconditioner = zeros([len(colNormSq), len(colNormSq)])
    for j in 0..<len(colNormSq):
      Preconditioner[j, j] = 1.0 / colNormSq[j]

  ZTZ = matmul(Z.T, Z)
  # start optimization
  predict(yPredQuad, yPredLinear, K, X, cfm.P, cfm.lams, cfm.w,
            cfm.intercept, ignoreDiag)
  residual = y - yPredQuad - yPredLinear
  lossOld = norm(residual, 2)^2 / float(nSamples)
  var stepsize = 0.0
  var nTol = 0
  self.it = 0
  for it in 0..<self.maxIter:
    if not self.optimal and self.it >= cfm.maxComponents:
      break
    
    predict(yPredQuad, yPredLinear, K, X, cfm.P, cfm.lams, cfm.w,
            cfm.intercept, ignoreDiag)
    residual = y - yPredQuad - yPredLinear
    
    # fit P
    var gradJ = matmul(X.T*residual, X)
    if ignoreDiag:
      for i in 0..<nSamples:
        for j in 0..<nFeatures:
          gradJ[j, j] -= residual[i] * X[i, j]^2

    let (_, p) = powerMethod(gradJ, self.maxIterPower, tol=self.tolPower)

    # Add or replace?
    var s = self.it
    if s >= cfm.maxComponents: # replace old p
      s = argmin(cfm.lams) # replace old p whose lambda is minimum
    cfm.P[s] = p

    for i in 0..<nSamples:
      if ignoreDiag:
        cacheK[i] = anovaSlow(X, cfm.P, i, 2, s, nFeatures, 0)
      else:
        cacheK[i] = polySlow(X, cfm.P, i, 2, s, nFeatures, 0)
    K[s] = cacheK

    # compute yPredQuad
    predict(yPredQuad, yPredLinear, K, X, cfm.P, cfm.lams, cfm.w,
            cfm.intercept, ignoreDiag)
    residual = y - yPredQuad - yPredLinear

    # compute stepsize
    if self.optimal:
      let d = self.eta * K[s] - yPredQuad
      stepsize = dot(d, residual) / norm(d, 2)^2
      stepsize = min(max(1e-10, stepsize), 1.0)
    else:
      stepsize = 2.0 / (float(self.it)+2.0)

    # update lams
    cfm.lams *= (1-stepsize)
    cfm.lams[s] += self.eta*stepsize
    if sum(cfm.lams) > self.eta:
      cfm.lams *= self.eta  / sum(cfm.lams)

    # fit w
    predict(yPredQuad, yPredLinear, K, X, cfm.P, cfm.lams, cfm.w,
            cfm.intercept, ignoreDiag)
    residual = y - yPredQuad
    yPredLinear <- 0.0
    if fitLinear:
      vmmul(residual, Z, resZ)
      let maggrad = norm(resZ, 1)
      let tolCG = 1e-5 * maggrad
      w *= colNormSq # since we use left-right preconditioning
      cg(ZTZ, resZ, w, maxIter=1000, init=false, tol=tolCG, 
         preconditioner=Preconditioner)
      cfm.w = w[0..<nFeatures]
      if fitIntercept:
        cfm.intercept = w[^1]
    elif fitIntercept:
      cfm.intercept = sum(residual) / float(nSamples)
    
    # stopping criterion
    predict(yPredQuad, yPredLinear, K, X, cfm.P, cfm.lams, cfm.w,
            cfm.intercept, ignoreDiag)
    residual = y - yPredQuad - yPredLinear
    lossNew = norm(residual, 2)^2  / float(nSamples)
    if lossOld - lossNew < self.tol:
      inc(nTol)
      if nTol == self.nTol:
        break
    else:
      nTol = 0
    lossOld = lossNew
    inc(self.it)
