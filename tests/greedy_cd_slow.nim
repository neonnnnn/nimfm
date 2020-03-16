import nimfm/loss, nimfm/tensor, nimfm/optimizers/optimizer_base
from nimfm/fm_base import checkTarget, checkInitialized
import sequtils, math, strformat, strutils
import fit_linear_slow, cfm_slow, kernels_slow

type
  GCDSlow* = ref object of BaseCSCOptimizer
    ## Greedy coordinate descent solver for convex factorization machines.
    ## In this solver, the regularization for interaction is not 
    ## squared Frobenius norm for P but the trace norm for interaction weight
    ## matrix.
    maxIterInner: int
    maxIterPower: int
    nRefitting: int
    fullyRefit: bool
    tolPower: float64


proc predict(P: Matrix, w, lams: Vector, intercept: float64, X: Matrix,
             yPred: var seq[float64], K: Matrix, degree: int) =
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    nComponents = P.shape[0]
  for i in 0..<nSamples:
    yPred[i] = intercept

  for j in 0..<nFeatures:
    for i in 0..<nSamples:
      yPred[i] += w[j] * X[i, j]

  for s in 0..<nComponents:
    if lams[s] == 0: continue
    for i in 0..<nSamples:
      yPred[i] += lams[s] * K[s, i]


proc newGCDSlow*(
  maxIter = 100, maxIterInner=100, nRefitting=10, fullyRefit=false,
  verbose = 2, tol = 1e-7, maxIterPower = 100, tolPower = 1e-6): GCDSlow =
  result = GCDSlow(
    maxIter: maxIter, maxIterInner: maxIterInner, nRefitting: nRefitting,
    fullyRefit: fullyRefit, tol: tol, verbose: verbose,
    maxIterPower: maxIterPower, tolPower: tolPower)


proc fitLams(lams: var Vector, s: int, beta: float64,
             K: Matrix, dL: Vector, mu: float64) {.inline.} =
  let nSamples = K.shape[1]
  var
    update = 0.0
    invStepSize = 0.0
    norm = 0.0
  for i in 0..<nSamples:
    update += dL[i] * K[s, i]
    norm += K[s, i]^2
  invStepSize = mu * norm
  lams[s] -= update / invStepSize
  # soft thresholding
  if (lams[s] - beta / invStepSize) > 0:
    lams[s] -= beta/invStepSize
  elif (lams[s] + beta / invStepSize) < 0:
    lams[s] += beta/invStepSize
  else:
    lams[s] = 0


proc refitDiag(X: Matrix, y: seq[float64], yPred: var seq[float64], P: Matrix,
               lams: var Vector, beta: float64, loss: LossFunction, K: Matrix, 
               dL: var Vector, w: Vector, intercept: float64): int =
  let nSamples = len(y)
  result = 0
  for s in 0..<len(lams):
    if lams[s] != 0.0:
      for i in 0..<nSamples:
        dL[i] = loss.dloss(y[i], yPred[i])
      fitLams(lams, s, beta, K, dL, loss.mu)
      predict(P, w, lams, intercept, X, yPred, K, 2)
      if lams[s] != 0: result += 1


proc fitZ(self: GCDSlow, X: Matrix, y: seq[float64],
          yPred: var seq[float64], P: var Matrix, lams: var Vector,
          beta: float64, loss: LossFunction, K: var Matrix,
          p, dL: var Vector, maxComponents, verbose: int,
          ignoreDiag: bool, XTRX: var Matrix,
          w: Vector, intercept: float64): float64 =
  var nComponents = 0
  let nSamples = X.shape[0]
  let nFeatures = X.shape[1]
  var lossOld = 0.0
  var addBase = false
  var evalue = 0.0
  var sNew: int
  for s in 0..<len(lams):
    if lams[s] != 0.0: nComponents += 1

  for i in 0..<nSamples:
    lossOld += loss.loss(y[i], yPred[i])
  for s in 0..<nComponents:
    lossOld += beta * abs(lams[s])
  lossOld /= float(nSamples)

  for it in 0..<self.maxIterInner:
    result = 0.0
    addBase = false
    predict(P, w, lams, intercept, X, yPred, K, 2)
    # add new base (dominate eigenvector)
    if nComponents < maxComponents:
      # compute XTRX
      for j1 in 0..<nFeatures:
        for j2 in 0..<nFeatures:
          XTRX[j1, j2] = 0.0
      for i in 0..<nSamples:
        dL[i]= loss.dloss(y[i], yPred[i])
        for j1 in 0..<nFeatures:
          for j2 in 0..<nFeatures:
            XTRX[j1, j2] += dL[i] * X[i, j1] * X[i, j2]
      if ignoreDiag:
        for i in 0..<nSamples:
          for j in 0..<nFeatures:
            XTRX[j, j] -= dL[i] * X[i, j]^2
        for j1 in 0..<nFeatures:
          for j2 in 0..<nFeatures:
            XTRX[j1, j2] *= 0.5
      # compute dominate eigen vector
      (evalue, p) = powerIteration(XTRX, self.maxIterPower, self.tol)
      # determine the row index and substitute
      for s in 0..<len(lams):
        if lams[s] == 0.0:
          sNew = s
          break
      for j in 0..<nFeatures:
        P[sNew, j] = p[j]
      for i in 0..<nSamples:
        if ignoreDiag:
          K[sNew, i] = anovaSlow(X, P, i, 2, sNew, nFeatures, 0)
        else:
          K[sNew, i] = polySlow(X, P, i, 2, sNew, nFeatures, 0)
      # fit lams[sNew]
      fitLams(lams, sNew, beta, K, dL, loss.mu)
      predict(P, w, lams, intercept, X, yPred, K, 2)
      if lams[sNew] != 0.0:
        nComponents += 1
        addBase = true
    
    # refitting
    if (it+1) mod self.nRefitting == 0:
      # diagonal refitting
      if not self.fullyRefit:
        nComponents = refitDiag(X, y, yPred, P, lams, beta, loss, K, dL,
                                w, intercept)
      else:
        raise newException(ValueError, "Not implemented, ToDo.")
    # stopping criterion
    if addBase or (it+1) mod self.nRefitting == 0:
      # compute objective for stopping criterion
      result = 0.0
      for s in 0..<len(lams):
        result += abs(lams[s])
      result *= beta / float(nSamples)
      for i in 0..<nSamples:
        result += loss.loss(y[i], yPred[i])
      result /= float(nSamples)
      
      # stopping criterion
      if abs(result - lossOld) < self.tol:
        break
      lossOld = result
  

proc fit*(self: GCDSlow, X: Matrix, y: seq[float64],
          cfm: var CFMSlow) =
  ## Fits the factorization machine on X and y by coordinate descent.
  cfm.init(X)
  let y = checkTarget(cfm, y)
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    maxComponents = cfm.maxComponents
    alpha0 = cfm.alpha0 * float(nSamples)
    alpha = cfm.alpha * float(nSamples)
    beta = cfm.beta * float(nSamples)
    fitLinear = cfm.fitLinear
    fitIntercept = cfm.fitIntercept
    loss = newLossFunction(cfm.loss)

  # caches
  var
    yPred = newSeqWith(nSamples, 0.0)
    A: Matrix = zeros([nSamples, 3])
    K: Matrix = zeros([cfm.maxComponents, nSamples])
    p: Vector = zeros([nFeatures])
    dL: Vector = zeros([nSamples])
    XTRX: Matrix = zeros([nFeatures, nFeatures])
    colNormSq: Vector = zeros([nFeatures])
    isConverged = false
    lossOld = 0.0
    lossNew = 0.0
  
  # init caches
  for i in 0..<nSamples:
    A[i, 0] = 1.0
  if fitLinear:
   for i in 0..<nSamples:
    for j in 0..<nFeatures:
       colNormSq[j] += X[i, j]^2

  # compute prediction
  for s in 0..<len(cfm.lams):
    if cfm.lams[s] != 0.0:
      for i in 0..<nSamples:
        if cfm.ignoreDiag:
          K[s, i] = anovaSlow(X, cfm.P, i, 2, s, nFeatures, 0)
        else:
          K[s, i] = polySlow(X, cfm.P, i, 2, s, nFeatures, 0)
  predict(cfm.P, cfm.w, cfm.lams, cfm.intercept, X, yPred, K, 2)
  # compute loss
  for i in 0..<nSamples:
    lossOld += loss.loss(y[i], yPred[i])
  if fitLinear:
    lossOld += alpha * norm(cfm.w, 2)^2
  if fitIntercept:
    lossOld += alpha0 * cfm.intercept^2
  lossOld /= float(nSamples)

  # start iteration
  for it in 0..<self.maxIter:
    lossNew = 0.0
    
    if fitIntercept:
      discard fitInterceptCD(cfm.intercept, y, yPred, nSamples, alpha0, loss)
      lossNew += alpha0 * cfm.intercept^2
    
    if fitLinear:
      discard fitLinearCD(cfm.w, X, y, yPred, colNormSq, alpha, loss)
      lossNew += alpha0 * norm(cfm.w, 2)^2

    lossNew /= float(nSamples)

    lossNew += fitZ(self, X, y, yPred, cfm.P, cfm.lams, beta, loss, K, p, dL,
                    maxComponents, self.verbose, cfm.ignoreDiag, XTRX,
                    cfm.w, cfm.intercept)
    
    if abs(lossNew - lossOld) < self.tol:
      isConverged = true
      break
    lossOld = lossNew