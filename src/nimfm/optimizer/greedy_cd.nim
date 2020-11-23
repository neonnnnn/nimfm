import ../dataset, ../tensor, ../kernels, ../extmath, ../loss
import ../model/convex_factorization_machine
from ../model/fm_base import checkTarget, checkInitialized
import fit_linear, optimizer_base, utils
import sequtils, math, strformat, strutils, sugar

type
  GreedyCD*[L] = ref object of BaseCSCOptimizer
    ## Greedy coordinate descent solver for convex factorization machines.
    ## In this solver, the regularization for P is not 
    ## squared Frobenius norm but the trace norm for interaction weight
    ## matrix (weight matrix for quadratic term).
    loss*: L
    maxIterInner: int
    maxIterPower: int
    nRefitting: int
    refitFully: bool
    tolPower: float64
    sigma: float64
    maxIterADMM: int
    tolADMM: float64
    maxIterLineSearch: int


proc newGreedyCD*[L](
  maxIter = 10, alpha0=1e-6, alpha=1e-3, beta=1e-5, loss: L=newSquared(),
  maxIterInner=10, nRefitting=10, refitFully=false,
  verbose = 1, tol = 1e-7, maxIterPower = 100, tolPower = 1e-7,
  sigma=1e-4, maxIterADMM=100, tolADMM=1e-4,
  maxIterLineSearch=100): GreedyCD[L] =
  ## Creates new GreedyCD for ConvexFactorizationMachine.
  ## maxIter: Maximum number of iteration for alternative optimization.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for second-order weights (trace norm).
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## maxIterInner: Maximum number of iteration for optimizing Z (P and lambda).
  ## maxIterPower: Maximum number of iteration for power iteration.
  ## nRefitting: Frequency of the refitting lams and P in inner loop.
  ##             If nRefitting > maxIterInner, lams and P are not reffitted.
  ## refitFully: Whether refit both P and lams or only lams.
  ## verbose: Whether to print information on optimization processes.
  ##          0: no printing.
  ##          1: prints the value of objective function at outer loop.
  ##          2: prints the value of objective function at innter loop.
  ##          3: prints the value of objective function at ADMM loop
  ##             (for refitFully=True).
  ## tol: Tolerance hyperparameter for stopping criterion.
  ##      If the decreasing of the objective value is smaller than tol,
  ##      then this solver stops.
  ## tolPower: Tolerance hyperparameter for stopping criterion 
  ##           of power iteration.
  ## sigma: Parameter for line search in fully refitting.
  ## maxIterADMM: Maximum number of iteration of ADMM in fully refitting.
  ## tolADMM: Tolerance hyperparameter of stopping criterion of ADMM 
  ##          in fully refitting.
  ## maxIterLineSearch: Maximum number of interation of line seaerch 
  ##                    in fully refitting.
  result = GreedyCD[L](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, loss: loss,
    maxIterInner: maxIterInner, nRefitting: nRefitting,
    refitFully: refitFully, tol: tol, verbose: verbose,
    maxIterPower: maxIterPower, tolPower: tolPower, sigma: sigma,
    maxIterADMM: maxIterADMM, tolADMM: tolADMM,
    maxIterLineSearch: maxIterLineSearch)


proc objectiveADMM[L](y, yPred: Vector, A, B, M: Matrix, rho: float, 
                      loss: L): float64 =
  for i in 0..<len(y):
    result += loss.loss(y[i], yPred[i])
  result += 0.5 * rho * norm(A-B+M, 2)^2


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
  # Soft-thresholding
  if (lams[s] - beta / invStepSize) > 0:
    lams[s] -= beta/invStepSize
  elif (lams[s] + beta / invStepSize) < 0:
    lams[s] += beta/invStepSize
  else:
    lams[s] = 0


proc refitDiag[L](y: seq[float64], yPred: var seq[float64], beta: float64, 
                  loss: L, K: Matrix, lams, dL: var Vector): int =
  let nSamples = len(y)
  result = 0
  for s in 0..<len(lams):
    if lams[s] != 0.0:
      for i in 0..<nSamples:
        dL[i] = loss.dloss(y[i], yPred[i])
      var old_lam = lams[s]
      fitLams(lams, s, beta, K, dL, loss.mu)
      yPred -= old_lam * K[s]
      yPred += lams[s] * K[s]
      if lams[s] != 0: result += 1


proc runADMM[L](self: GreedyCD, y: Vector, yPred: var Vector, X: ColDataset,
                beta: float64, loss: L, P, K: Matrix, lams, dL: var Vector,
                ignoreDiag: bool, A, B, M: var Matrix, 
                nSamples, nFeatures, nComponents: int): void = 
  var vecdA = zeros([nComponents^2])
  var vecD = zeros([nComponents^2])
  var D = zeros([nComponents, nComponents])
  var AP = matmul(A, P)
  var DP = zeros([nComponents, nFeatures])
  var PXT = zeros([nComponents, nSamples])
  var DPXT = zeros([nComponents, nSamples])
  var diagPTDP = zeros([nFeatures])
  var pi = zeros([nSamples])
  var ev = zeros([nComponents])
  var yPredQuad = zeros([nSamples])
  var yPredDiff = zeros([nSamples])
  var delta: Vector
  var rho = 1.0
  let PT = P.T

  if ignoreDiag:
    delta = zeros([nFeatures])

  # compute cache PXT
  PXT[0..^1, 0..^1] = 0.0
  for s in 0..<nComponents:
    linear(X, P[s], PXT[s]) # O(nComponents * nnz(X))

  # Linear operator for newton-cg
  proc linearOp(p: Vector, result: var Vector): void =
    # substitute D
    for s1 in 0..<nComponents:
      for s2 in 0..<nComponents:
        D[s1, s2] = p[s1+s2*nComponents]
    # compute pi
    matmul(D, P, DP) # O(nComponents^2 * nFeatures)
    for s in 0..<nComponents:
      linear(X, DP[s], DPXT[s]) # O(nComponents * nnz(X))
    pi = sum(DPXT*PXT, axis=0) # O(nComponents * nSamples)
    if ignoreDiag:
      diagPTDP = sum(P*DP, axis=0) # O(nComponents * nFeatures)
      for j in 0..<nFeatures: # O(nnz(X))
        for (i, val) in X.getCol(j):
          pi[i] -= val * val * diagPTDP[j]
      pi *= 0.5

    # result = vec(matmul(PXT*pi, PXT.T)) # O(nComponents^2 * nSamples), slow
    # Fast (?): O(nComponents * nnz(X) + nComponents^2 nFeatures)
    result = vec(matmul(matmul(PXT*pi, X), PT)) 
    if ignoreDiag:
      delta[0..^1] = 0.0 # = vmmul(pi, X*X)
      for j in 0..<nFeatures: # O(nnz(X))
        for (i, val) in X.getCol(j):
          delta[j] += pi[i] * val * val
      result -= vec(matmul(P*delta, PT)) # O(nComponents^2 * nFeatures)
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


  # Perform ADMM
  for itADMM in 0..<self.maxIterADMM:
    # optimize A
    # compute gradient wrt A, vecdA
    for i in 0..<nSamples: dL[i] = loss.dloss(y[i], yPred[i])
    vecD[0..^1] = 0.0
    # vecdA = vec(matmul(PXT*dL, PXT.T)+rho*(A-B+M)) # O(n*k^2), slow
    # Fast (?): O(nComponents * nnz(X) + nComponents^2 nFeatures)
    vecdA = vec(matmul(matmul(PXT*dL, X), PT) + rho*(A-B+M)) 
    if ignoreDiag:
      delta[0..^1] = 0.0 # vmmul(dL, X*X)
      for j in 0..<nFeatures: # O(nnz(X))
        for (i, val) in X.getCol(j):
          delta[j] += dL[i] * val * val
      vecdA -= vec(matmul(P*delta, PT)) # O(nComponents^2 * nFeatures)
      vecdA *= 0.5
    
    # compute update direction D and recompute DP
    let maggrad = norm(vecdA, 1)
    let tolCG = 1e-3 * maggrad
    # cg(linearOp, vecdA, vecD, maxIter=200, tol=tolCG)
    cg(linearOp, vecdA, vecD, maxIter=200, tol=tolCG, 
       preconditioner=preconditioner)
    for s1 in 0..<nComponents:
      for s2 in 0..<nComponents:
        D[s1, s2] = vecD[s1+s2*nComponents]
    matmul(D, P, DP)
   
    # compute quadratic term wrt D
    DPXT[0..^1, 0..^1] = 0.0
    for s in 0..<nComponents:
      linear(X, DP[s], DPXT[s])
    yPredDiff = sum(DPXT * PXT, axis=0)
    if ignoreDiag:
      diagPTDP = sum(P*DP, axis=0)
      for j in 0..<nFeatures:
        for (i, val) in X.getCol(j):
          yPredDiff[i] -= val * val * diagPTDP[j]
      yPredDiff *= 0.5
    
    # perform line search
    var eta = 1.0
    let condition = - self.sigma * dot(vecdA, vecD)
    let objOld = objectiveADMM(y, yPred, A, B, M, rho, loss)
    yPred -= eta * yPredDiff
    A -= eta * D
    for itSearch in 0..<self.maxIterLineSearch:
      let objNew = objectiveADMM(y, yPred, A, B, M, rho, loss)
      if (objNew - objOld) < eta * condition: break
      eta *= 0.5
      # yPred += 2 * eta * yPredDiff - eta * yPredDiff
      yPred += eta * yPredDiff
      A += eta * D
    # line search done, update A, AP, and yPred
    yPredQuad += eta * yPredDiff
    matmul(A, P, AP)

    # optimize B
    D = B # D = old B
    B = A + M
    dsyev(B, ev, columnWise=true)
    for s in 0..<nComponents: # soft-thresholding wrt eigenvalues
      ev[s] = float64(sgn(ev[s]))*max(0.0, abs(ev[s]) - beta / rho)
    B = matmul(B*ev, B.T)

    # optimize M
    M += rho * A
    M -= rho * B

    # stopping criterion
    let primalResidual = norm(A-B, 2)
    let dualResidual = rho * norm(B-D, 2)
    if primalResidual < self.tolADMM and dualResidual < self.tolADMM: break
    
    # adapt rho
    if primalResidual > 10.0 * dualResidual: rho *= 2.0
    elif dualResidual > 10.0 * primalResidual: rho /= 2.0
    if self.verbose > 2:
      stdout.write(fmt"    Iteration (ADMM): {itADMM}")
      stdout.write(fmt"  Primal Residual: {primalResidual:1.4e}")
      stdout.write(fmt"  Dual Residual: {dualResidual:1.4e}")
      stdout.write("\n")
      stdout.flushFile()

  yPred += yPredQuad


proc refitFully[L](self: GreedyCD, y: Vector, yPred: var Vector, X: ColDataset,
                   beta: float64, loss: L, P, K, cacheK: var Matrix, 
                   lams, dL: var Vector, ignoreDiag: bool): int = 
  var nComponents = 0
  # Squeeze P and lams
  let nSamples = len(y)
  let nFeatures = X.nFeatures
  for s in 0..<len(lams):
    if lams[s] != 0:
      lams[nComponents] = lams[s]
      P[nComponents] = P[s]
      inc(nComponents)
      yPred -= lams[s] * K[s]
  # unused variables set to be zeros
  for s in nComponents..<len(lams):
    lams[s] = 0.0
    P[s, 0..^1] = 0.0
    K[s, 0..^1] = 0.0
  
  var A = eye(nComponents)
  var B = eye(nComponents)
  var M = zeros([nComponents, nComponents])

  orthogonalize(P, nComponents)
  # recompute Kernels
  for s in 0..<nComponents:
    if ignoreDiag: anova(X, P, cacheK, 2, s)
    else: poly(X, P, cacheK, 2, s)
    K[s] = cacheK[0..^1, 2]
    yPred += K[s]
  
  # optimize A, B, and M
  runADMM(self, y, yPred, X, beta, loss, P[0..<nComponents], K, lams, dL,
          ignoreDiag, A, B, M, nSamples, nFeatures, nComponents)
  # finalize A
  var ev = zeros([nComponents])
  dsyev(A, ev, columnWise=false)
  result = int(norm(ev, 0))
  P[0..<nComponents] = matmul(A, P[0..<nComponents])
  lams[0..<nComponents] = ev

  # synchronize predictions
  for s in 0..<nComponents:
    yPred -= K[s]
    K[s, 0..^1] = 0.0
  for s in 0..<result:
    if ignoreDiag: anova(X, P, cacheK, 2, s)
    else: poly(X, P, cacheK, 2, s)
    K[s] = cacheK[0..^1, 2]
    yPred += K[s] * lams[s]


proc fitZ[L](self: GreedyCD, X: ColDataset, y: seq[float64],
             yPred: var seq[float64], P: var Matrix, lams: var Vector,
             beta: float64, loss: L, cacheK: var Matrix, K: var Matrix,
             maxComponents, verbose: int, ignoreDiag: bool) =
  var nComponents = 0
  let nSamples = X.nSamples
  let nFeatures = X.nFeatures
  var oldLossVal = 0.0
  var addBase = false
  var dL = zeros([nSamples])
  var Xp = zeros([nSamples])
  var cache = zeros([nSamples])
  nComponents = int(norm(lams, 0))
  for i in 0..<nSamples:
    oldLossVal += loss.loss(y[i], yPred[i])
  oldLossVal += beta * norm(lams, 1)
  oldLossVal /= float(nSamples)
  
  proc linearOpPower(p: Vector, result: var Vector) =
    mvmul(X, p, Xp)
    Xp *= dL
    vmmul(Xp, X, result)
    if ignoreDiag:
      for j in 0..<nFeatures:
        for (i, val) in X.getCol(j):
          result[j] -= val*val*dL[i]*p[j]

  for it in 0..<self.maxIterInner:
    addBase = false
    # Add a new vector to basis (dominate eigenvector)
    if nComponents < maxComponents:
      # Compute dL = residual for squared loss
      for i in 0..<nSamples:
        dL[i] = loss.dloss(y[i], yPred[i])
      let (_, p) = powerMethod(linearOpPower, nFeatures, self.maxIterPower, 
                               self.tolPower)
      # Add a new row or re-use existing row, which?
      var sNew = len(lams)
      for s in 0..<len(lams):
        if lams[s] == 0.0: # re-use existing row
          sNew = s
          break
      # Add a new row
      if sNew == len(lams):
        lams.add(0.0)
        P.addRow(p)
      else: # Re-use
        P[sNew] = p
      # Compute kernel
      if ignoreDiag: anova(X, P, cacheK, 2, sNew)
      else: poly(X, P, cacheK, 2, sNew)
      cache = cacheK[0..^1, 2]
      if sNew == len(K): # Add new row
        K.addRow(cache)
      else: # Re-use
        K[sNew] = cache
      # Fit lams[sNew]
      fitLams(lams, sNew, beta, K, dL, loss.mu)
      if lams[sNew] != 0.0:
        yPred += lams[sNew] * K[sNew]
        nComponents += 1
        addBase = true
    
    # Refitting
    if (it+1) mod self.nRefitting == 0:
      # diagonal refitting
      if not self.refitFully:
        nComponents = refitDiag(y, yPred, beta, loss, K, lams, dL)
      elif nComponents != 0:
        nComponents = refitFully(self, y, yPred, X, beta, loss, P, K,
                                 cacheK, lams, dL, ignoreDiag)
    # if basis is changed or refitted
    if addBase or (it+1) mod self.nRefitting == 0 or it == self.maxIterInner-1:
      # compute objective for stopping criterion
      var newLossVal = norm(lams, 1) * beta
      for i in 0..<nSamples:
        newLossVal += loss.loss(y[i], yPred[i])
      newLossVal /= float(nSamples)
      
      if verbose > 1:
        let iterAligned =  align($(it+1), len($self.maxIterInner))
        stdout.write(fmt"   Iteration: {iterAligned}")
        stdout.write(fmt"   Objective: {newLossVal:1.4e}")
        stdout.write(fmt"   Decreasing: {oldLossVal - newLossVal:1.4e}")
        stdout.write("\n")
        stdout.flushFile()

      # Stopping criterion
      if abs(newLossVal - oldLossVal) < self.tol:
        if verbose > 1:
          echo("   Converged at iteration ", it+1, ".")
        break
      oldLossVal = newLossVal
  

proc fit*[L](self: GreedyCD[L], X: ColDataset, y: seq[float64],
             cfm: ConvexFactorizationMachine,
             callback: (GreedyCD[L], ConvexFactorizationMachine)->void=nil) =
  ## Fits the factorization machine on X and y by coordinate descent.
  cfm.init(X)
  let y = checkTarget(cfm, y)
  let
    nSamples = X.nSamples
    maxComponents = cfm.maxComponents
    alpha0 = self.alpha0 * float(nSamples)
    alpha = self.alpha * float(nSamples)
    beta = self.beta * float(nSamples)
    fitLinear = cfm.fitLinear
    fitIntercept = cfm.fitIntercept

  # caches
  var
    yPred = newSeqWith(nSamples, 0.0)
    cacheK: Matrix = zeros([nSamples, 3])
    K: Matrix = zeros([len(cfm.lams), nSamples])
    colNormSq: Vector
    isConverged = false
    oldLossVal, newLossVal, oldRegVal, newRegVal: float64

  # init caches
  cacheK[0..^1, 0] = 1.0
  if fitLinear:
    colNormSq = norm(X, p=2, axis=0)
    colNormSq *= colNormSq

  # compute prediction
  linear(X, cfm.w, yPred)
  yPred += cfm.intercept
  for s in 0..<len(cfm.lams):
    if cfm.ignoreDiag: anova(X, cfm.P, cacheK, 2, s)
    else: poly(X, cfm.P, cacheK, 2, s)
    K[s] = cacheK[0..^1, 2]
    yPred += cfm.lams[s] * K[s]
  
  # compute loss
  (oldLossVal, oldRegVal) = objective(y, yPred, cfm.P, cfm.w, cfm.intercept,
                                      self.alpha0, self.alpha, 0, self.loss)
  oldRegVal += self.beta * norm(cfm.lams, 1) # trace norm

  # start iteration
  for it in 0..<self.maxIter:
    if self.verbose > 0:
      echo(fmt"Outer Iteration {it+1}")
    
    if fitIntercept:
      discard fitInterceptCD(cfm.intercept, y, yPred, nSamples, alpha0, 
                             self.loss)
    
    if fitLinear:
      discard fitLinearCD(cfm.w, X, y, yPred, colNormSq, alpha, self.loss)

    fitZ(self, X, y, yPred, cfm.P, cfm.lams, beta, self.loss, cacheK,
         K, maxComponents, self.verbose, cfm.ignoreDiag)
    
    (newLossVal, newRegVal) = objective(y, yPred, cfm.P, cfm.w, cfm.intercept,
                                        self.alpha0, self.alpha, 0, self.loss)
    newRegVal += self.beta * norm(cfm.lams, 1)
    
    if not callback.isNil:
      callback(self, cfm)
    if self.verbose > 0:
      stdout.write(fmt"   Loss: {newLossVal:1.4e}")
      stdout.write(fmt"   Reg: {newRegVal:1.4e}")
      stdout.write("\n")
    if abs(newLossVal + newRegVal - oldLossVal - oldRegVal) < self.tol:
      if self.verbose > 0:
        echo("Converged at iteration ", it+1, ".")
      isConverged = true
      break
    oldLossVal = newLossVal
    oldRegVal = newRegVal

    # Re-compute predictons for next iter (in O(nnz(X)k))
    if it < self.maxIter - 1:
      linear(X, cfm.w, yPred)
      yPred += cfm.intercept
      for s in 0..<len(cfm.lams):
        yPred += cfm.lams[s] * K[s]

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")