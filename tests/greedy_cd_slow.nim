import nimfm/tensor, nimfm/optimizers/optimizer_base
from nimfm/fm_base import checkTarget, checkInitialized
import sequtils, math
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
    refitFully: bool
    tolPower: float64
    sigma: float64
    maxIterADMM: int
    tolADMM: float64
    maxIterLineSearch: int


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
    if lams[s] != 0.0:
      for i in 0..<nSamples:
        yPred[i] += lams[s] * K[s, i]


proc newGCDSlow*(
  maxIter = 100, maxIterInner=10, nRefitting=10, refitFully=false,
  verbose = 2, tol = 1e-7, maxIterPower = 200, tolPower = 1e-7,
  sigma=1e-4, maxIterADMM=100, tolADMM=1e-4, maxIterLineSearch=100): GCDSlow =
  result = GCDSlow(
    maxIter: maxIter, maxIterInner: maxIterInner, nRefitting: nRefitting,
    refitFully: refitFully, tol: tol, verbose: verbose,
    maxIterPower: maxIterPower, tolPower: tolPower, sigma: sigma,
    maxIterADMM: maxIterADMM, tolADMM: tolADMM, 
    maxIterLineSearch: maxIterLineSearch)


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


proc refitDiag[L](X: Matrix, y: seq[float64], yPred: var seq[float64], P: Matrix,
                  lams: var Vector, beta: float64, loss: L, K: Matrix, 
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


proc refitFully[L](self: GCDSlow, X: Matrix, y: seq[float64],
                   yPred: var seq[float64], P: var Matrix,
                   lams: var Vector, beta: float64, loss: L, K: Matrix, 
                   dR: var Vector, w: Vector, intercept: float64,
                   ignoreDiag: bool): int =
  let nSamples = len(y)
  var nComponents = 0
  # Squeeze P and lams
  let nFeatures = X.shape[1]
  for s in 0..<len(lams):
    if lams[s] != 0:
      lams[nComponents] = lams[s]
      for j in 0..<nFeatures:
        P[nComponents, j] = P[s, j]
      inc(nComponents)
  for s in nComponents..<len(lams):
    lams[s] = 0.0
    P[s, 0..^1] = 0.0
  var A = eye(nComponents)
  var D = zeros([nComponents, nComponents])
  var B = eye(nComponents)
  var M = zeros([nComponents, nComponents])
  var ev = zeros([nComponents])
  var AP = zeros([nComponents, nFeatures])
  var DP = zeros([nComponents, nFeatures])
  var vecD = ones([nComponents^2])
  var vecdA = zeros([nComponents^2])
  var delta = zeros([nFeatures])
  var yPredDiff = zeros([nSamples])
  var pi = zeros([nFeatures])
  var rho = 1.0
  # preprocessing
  orthogonalize(P, nComponents)
  let P2 = P[0..<nComponents]
  let PXT: Matrix =  matmul(P2, X.T)
  matmul(A, P2, AP)
  # compute predictions
  yPred = sum(matmul(AP, X.T) * PXT, axis=0)
  if ignorediag:
    yPred -= mvmul(X*X, sum(P*AP, axis=0))
    yPred *= 0.5
  yPred += mvmul(X, w)
  yPred += intercept

  proc linearOp(p: Vector, result: var Vector) =
    # substitute D
    for s1 in 0..<nComponents:
      for s2 in 0..<nComponents:
        D[s1, s2] = p[s1+s2*nComponents] 
    # compute DP
    matmul(D, P2, DP)
    pi = sum(matmul(DP, X.T)*PXT, axis=0)
    if ignoreDiag:
      pi -= mvmul(X*X, sum(P2*DP, axis=0))
      pi *= 0.5

    result = vec(matmul(PXT*pi, PXT.T))
    if ignoreDiag:
      result -= vec(matmul(P2*(vmmul(pi, X*X)), P2.T))
    result += rho * p

  # preconditioner for conjugate gradient
  var diagHesseApprox = zeros([nComponents^2])
  for s1 in 0..<nComponents: # O(nSamples * nComponents^2)
    for s2 in 0..<nComponents:
      for i in 0..<nSamples:
        diagHesseApprox[s1*nComponents+s2] = (PXT[s1, i]^2) * (PXT[s2, i]^2)
  diagHesseApprox *= loss.mu
  diagHesseApprox += 1e-5
  proc preconditioner(p: var Vector): void =
    p /= (diagHesseApprox + rho)

  # optimize
  for itADMM in 0..<self.maxIterADMM:
    # optimize A
    # compute deirvatives
    for i in 0..<nSamples: dR[i] = loss.dloss(y[i], yPred[i])
    # compute gradient wrt A
    vecD[0..^1] = 0.0
    vecdA = vec(matmul(PXT*dR, PXT.T)+rho*(A-B+M))
    if ignoreDiag:
      vmmul(dR, X*X, delta)
      vecdA -= vec(matmul(P2*delta, P2.T))
      vecdA *= 0.5
    let maggrad = norm(vecdA, 1)
    let tolCG = 1e-3 * maggrad
    cg(linearOp, vecdA, vecD, maxIter=200, tol=tolCG,
       preconditioner=preconditioner)
    # substitute D
    for s1 in 0..<nComponents:
      for s2 in 0..<nComponents:
        D[s1, s2] = vecD[s1+s2*nComponents]
    matmul(D, P2, DP)

    # line search
    var objOld = 0.0
    var objNew = 0.0
    for i in 0..<nSamples:
      objOld += loss.loss(y[i], yPred[i])
    objOld += 0.5 * rho * norm(A-B+M, 2)^2
    
    yPredDiff = sum(matmul(DP, X.T) * PXT, axis=0)
    if ignoreDiag:
      yPredDiff -= mvmul(X*X, sum(P*DP, axis=0))
      yPredDiff *= 0.5

    # perform line search
    var eta = 1.0
    let condition = - self.sigma * dot(vecdA, vecD)
    for itSearch in 0..<self.maxIterLineSearch:
      objNew = 0.0
      for i in 0..<nSamples:
        objNew += loss.loss(y[i], yPred[i] - eta*yPredDiff[i])
      objNew += 0.5 * rho * norm(A-eta*D-B+M, 2)^2
      if (objNew - objOld) < eta * condition: break
      eta *= 0.5
    # line search done, update A, AP, and yPred
    yPred -= eta * yPredDiff
    A -= eta * D
    matmul(A, P2, AP)

    # optimize B
    D = B # D = old B
    B = A + M
    dsyev(B, ev, columnWise=true)
    for s in 0..<nComponents:
      ev[s] = float64(sgn(ev[s]))*max(0.0, abs(ev[s]) - beta / rho)
    B = matmul(B*ev, B.T)
    # optimize M
    M += rho * A
    M -= rho * B

    # stopping criterion
    let primalResidual = norm(A-B, 2)
    let dualResidual = rho * norm(D-B, 2)
    if primalResidual < self.tolADMM and dualResidual < self.tolADMM: break
    
    if primalResidual > 10.0 * dualResidual: rho *= 2.0
    elif dualResidual > 10.0 * primalResidual: rho /= 2.0

  # finalize
  dsyev(A, ev, columnWise=false)
  result = int(norm(ev, 0))
  matmul(A, P2, AP)
  P[0..<nComponents] = AP
  lams[0..<result] = ev


proc fitZ[L](self: GCDSlow, X: Matrix, y: seq[float64],
             yPred: var seq[float64], P: var Matrix, lams: var Vector,
             beta: float64, loss: L, K: var Matrix,
             p, dL: var Vector, maxComponents, verbose: int,
             ignoreDiag: bool, XTRX: var Matrix,
             w: Vector, intercept: float64): float64 =
  var nComponents = 0
  let nSamples = X.shape[0]
  let nFeatures = X.shape[1]
  var objOld = 0.0
  var addBase = false
  var evalue = 0.0
  var sNew: int
  for s in 0..<len(lams):
    if lams[s] != 0.0: nComponents += 1

  for i in 0..<nSamples:
    objOld += loss.loss(y[i], yPred[i])
  for s in 0..<nComponents:
    objOld += beta * abs(lams[s])
  objOld /= float(nSamples)

  for it in 0..<self.maxIterInner:
    result = 0.0
    addBase = false
    predict(P, w, lams, intercept, X, yPred, K, 2)
    # add new base (dominate eigenvector)
    if nComponents < maxComponents:
      # compute XTRX
      XTRX[0..^1, 0..^1] = 0.0
      for i in 0..<nSamples:
        dL[i]= loss.dloss(y[i], yPred[i])
      matmul(X.T*dL, X, XTRX)

      if ignoreDiag:
        for i in 0..<nSamples:
          for j in 0..<nFeatures:
            XTRX[j, j] -= dL[i] * X[i, j]^2
        XTRX *= 0.5

      # compute dominate eigen vector
      (evalue, p) = powerMethod(XTRX, self.maxIterPower, self.tol)
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
      if not self.refitFully:
        nComponents = refitDiag(X, y, yPred, P, lams, beta, loss, K, dL,
                                w, intercept)
      else:
        nComponents = refitFully(self, X, y, yPred, P, lams, beta, loss, K, dL,
                                 w, intercept, ignoreDiag)
        for s in 0..<nComponents:
          for i in 0..<nSamples:
            if ignoreDiag: K[s, i] = anovaSlow(X, P, i, 2, s, nFeatures, 0)
            else: K[s, i] = polySlow(X, P, i, 2, s, nFeatures, 0)
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
      if abs(result - objOld) < self.tol:
        break
      objOld = result
  

proc fit*[L](self: GCDSlow, X: Matrix, y: seq[float64],
          cfm: var CFMSlow[L]) =
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
    loss = cfm.loss

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
    objOld = 0.0
    objNew = 0.0
  
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
    objOld += loss.loss(y[i], yPred[i])
  if fitLinear:
    objOld += alpha * norm(cfm.w, 2)^2
  if fitIntercept:
    objOld += alpha0 * cfm.intercept^2
  objOld /= float(nSamples)

  # start iteration
  for it in 0..<self.maxIter:
    objNew = 0.0
    
    if fitIntercept:
      discard fitInterceptCD(cfm.intercept, y, yPred, nSamples, alpha0, loss)
      objNew += alpha0 * cfm.intercept^2
    
    if fitLinear:
      discard fitLinearCD(cfm.w, X, y, yPred, colNormSq, alpha, loss)
      objNew += alpha0 * norm(cfm.w, 2)^2

    objNew /= float(nSamples)

    objNew += fitZ(self, X, y, yPred, cfm.P, cfm.lams, beta, loss, K, p, dL,
                    maxComponents, self.verbose, cfm.ignoreDiag, XTRX,
                    cfm.w, cfm.intercept)
    
    if abs(objNew - objOld) < self.tol:
      isConverged = true
      break
    objOld = objNew