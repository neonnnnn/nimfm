import ../loss, ../dataset, ../tensor, ../kernels
import ../convex_factorization_machine
from ../fm_base import checkTarget, checkInitialized
import fit_linear, optimizer_base
import sequtils, math, random, strformat, strutils

type
  GreedyCoordinateDescent* = ref object of BaseCSCOptimizer
    ## Greedy coordinate descent solver for convex factorization machines.
    ## In this solver, the regularization for interaction is not 
    ## squared Frobenius norm for P but the trace norm for interaction weight
    ## matrix.
    maxIterInner: int
    maxIterPower: int
    nRefitting: int
    fullyRefit: bool
    tolPower: float64


proc newGreedyCoordinateDescent*(
  maxIter = 100, maxIterInner=100, nRefitting=10, fullyRefit=false,
  verbose = 2, tol = 1e-7, maxIterPower = 100, tolPower = 1e-6):
     GreedyCoordinateDescent =
  ## Creates new GreedyCoordinateDescent for ConvexFactorizationMachine.
  ## maxIter: Maximum number of iteration for alternative optimization.
  ## maxIterInner: Maximum number of iteration for optimizing Z (P and lambda).
  ## maxIterPower: Maximum number of iteration for power iteration.
  ## nRefitting: Frequency of the refitting lams and P in inner loop.
  ##             If nRefitting > maxIterInner, lams and P are not reffitted.
  ## fullyRefit: Whether refit both P and lams or only lams.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyper-parameter for stopping criterion.
  ## tolPower: Tolerance hyper-parameter for stopping criterion 
  ## in power iteration.
  result = GreedyCoordinateDescent(
    maxIter: maxIter, maxIterInner: maxIterInner, nRefitting: nRefitting,
    fullyRefit: fullyRefit, tol: tol, verbose: verbose,
    maxIterPower: maxIterPower, tolPower: tolPower)


proc powerIteration(X: CSCDataset, dL: Vector, p, cache: var Vector,
                    maxIter=20, tol=1e-4, ignoreDiag: bool) =
  var pj = 0.0
  var norm = 0.0
  let nSamples = X.nSamples
  let nFeatures = X.nFeatures
  var ev, evOld: float64
  # init p
  for j in 0..<nFeatures:
    p[j] = 2*rand(1.0) - 1.0
  p /= norm(p, 2)
  # start power iteration
  for it in 0..<maxIter:
    # compute X^\top R X p
    # compute Xp
    linear(X, p, cache)
    # compute R X p
    for i in 0..<nSamples:
      cache[i] *= dL[i]
    ev = 0.0
    # compute X^\top R X p
    # compute eigen value
    for j in 0..<nFeatures:
      pj = 0.0
      for (i, val) in X.getCol(j):
        pj += val * cache[i]
        if ignoreDiag:
          pj -= dL[i]*p[j]*val^2
      if ignoreDiag:
        pj *= 0.5
      ev += pj*p[j]
      p[j] = pj
    p /= norm(p, 2)
    if it > 0 and abs(ev-evOld) < tol:
      break
    evOld = ev


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


proc refitDiag(y: seq[float64], yPred: var seq[float64], lams: var Vector,
               beta: float64, loss: LossFunction, K: Matrix,
               dL: var Vector): int =
  let nSamples = len(y)
  result = 0
  for s in 0..<len(lams):
    if lams[s] != 0.0:
      for i in 0..<nSamples:
        dL[i] = loss.dloss(y[i], yPred[i])
      var lamOld = lams[s]
      fitLams(lams, s, beta, K, dL, loss.mu)
      for i in 0..<nSamples:
        yPred[i] -= (lamOld-lams[s]) * K[s, i]

      if lams[s] != 0: result += 1


proc fitZ(self: GreedyCoordinateDescent, X: CSCDataset, y: seq[float64],
          yPred: var seq[float64], P: var Matrix, lams: var Vector,
          beta: float64, loss: LossFunction, A: var Matrix, K: var Matrix,
          p, dL, cache: var  Vector, maxComponents, verbose: int,
          ignoreDiag: bool): float64 =
  var nComponents = 0
  let nSamples = X.nSamples
  var lossOld = 0.0
  var addBase = false
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
    # add new base (dominate eigenvector)
    if nComponents < maxComponents:
      # compute dL = residual for squared loss
      for i in 0..<nSamples:
        dL[i] = loss.dloss(y[i], yPred[i])
      powerIteration(X, dL, p, cache, self.maxIterPower, self.tolPower, 
                     ignoreDiag)
      # add new row or re-use existing row, which?
      var sNew = len(lams)
      for s in 0..<len(lams):
        if lams[s] == 0: # re-use existing row
          sNew = s
          break
      # add new row
      if sNew == len(lams):
        lams.add(0)
        P.addRow(p)
      if ignoreDiag: anova(X, P, A, 2, sNew, 0)
      else: poly(X, P, A, 2, sNew, 0)

      for i in 0..<nSamples:
        cache[i] = A[i, 2]
      if sNew == len(K): # add new row
        K.addRow(cache)
      else: # re-use exiting row
        for i in 0..<nSamples:
          K[sNew, i] = cache[i]
      # fit lams[sNew]
      fitLams(lams, sNew, beta, K, dL, loss.mu)
      for i in 0..<nSamples:
        yPred[i] += lams[sNew] * K[sNew, i]
      if lams[sNew] != 0.0:
        nComponents += 1
        addBase = true
    
    # refitting
    if (it+1) mod self.nRefitting == 0:
      # diagonal refitting
      if not self.fullyRefit:
        nComponents = refitDiag(y, yPred, lams, beta, loss, K, dL)
      else:
        raise newException(ValueError, "Not implemented, ToDo.")
    
    if addBase or (it+1) mod self.nRefitting == 0:
      # compute objective for stopping criterion
      result = 0.0
      for s in 0..<len(lams):
        result += abs(lams[s])
      result *= beta / float(nSamples)
      for i in 0..<nSamples:
        result += loss.loss(y[i], yPred[i])
      result /= float(nSamples)
      
      if verbose > 1:
        stdout.write(
          fmt"   Iteration: {align($(it+1), len($self.maxIterInner))}"
        )
        stdout.write(fmt"   Objective: {result:1.4e}")
        stdout.write(fmt"   Dicreasing: {lossOld - result:1.4e}")
        stdout.write("\n")
        stdout.flushFile()

      # stopping criterion
      if abs(result - lossOld) < self.tol:
        if verbose > 1:
          echo("   Converged at iteration ", it+1, ".")
        break
      lossOld = result
  

proc fit*(self: GreedyCoordinateDescent, X: CSCDataset, y: seq[float64],
          cfm: var ConvexFactorizationMachine) =
  ## Fits the factorization machine on X and y by coordinate descent.
  cfm.init(X)
  let y = checkTarget(cfm, y)
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
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
    K: Matrix = zeros([len(cfm.lams), nSamples])
    p: Vector = zeros([nFeatures])
    dL: Vector = zeros([nSamples])
    cache: Vector = zeros([nSamples])
    colNormSq: Vector = zeros([nFeatures])
    isConverged = false
    lossOld = 0.0
    lossNew = 0.0
  
  # init caches
  for i in 0..<nSamples:
    A[i, 0] = 1.0
  if fitLinear:
   for j in 0..<nFeatures:
     for (_, val) in X.getCol(j):
       colNormSq[j] += val^2

  # compute prediction
  linear(X, cfm.w, yPred)
  for i in 0..<nSamples:
    yPred[i] += cfm.intercept
  for s in 0..<len(cfm.lams):
    if cfm.ignoreDiag: anova(X, cfm.P, A, 2, s, 0)
    else: poly(X, cfm.P, A, 2, s, 0)
    for i in 0..<nSamples:
      yPred[i] += cfm.lams[s] * A[i, 2]
      K[s, i] = A[i, 2]
  
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
    if self.verbose > 0:
      echo(fmt"Outer Iteration {it+1}")
    
    if fitIntercept:
      discard fitInterceptCD(cfm.intercept, y, yPred, nSamples, alpha0, loss)
      lossNew += alpha0 * cfm.intercept^2
    
    if fitLinear:
      discard fitLinearCD(cfm.w, X, y, yPred, colNormSq, alpha, loss)
      lossNew += alpha0 * norm(cfm.w, 2)^2

    lossNew /= float(nSamples)

    lossNew += fitZ(self, X, y, yPred, cfm.P, cfm.lams, beta, loss, A, K, p, dL,
                    cache, maxComponents, self.verbose, cfm.ignoreDiag)
    
    if self.verbose > 0:
      stdout.write(fmt"   Whole Objective: {lossNew:1.4e}")
      stdout.write("\n")
    if abs(lossNew - lossOld) < self.tol:
      if self.verbose > 0:
        echo("Converged at iteration ", it+1, ".")
      isConverged = true
      break
    lossOld = lossNew

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")