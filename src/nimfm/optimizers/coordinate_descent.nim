import ../loss, ../dataset, ../tensor, ../kernels, ../factorization_machine
import fit_linear, base
import sequtils, math, strformat, strutils

type
  CoordinateDescent* = ref object of BaseCSCOptimizer
    ## Coordinate descent solver.


proc newCoordinateDescent*(maxIter = 100, verbose = true, tol = 1e-3):
                           CoordinateDescent =
  ## maxIter: Maximum number of iteration. In one iteration. \
  ## all parameters are updated once by using all samples.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyper-parameter for stopping criterion.
  result = CoordinateDescent(maxIter: maxIter, tol: tol, verbose: verbose)


proc computeDerivative(dA: var Vector, A: var Matrix, psj, xij: float64,
                       i, degree: int) {.inline.} =
  dA[0] = xij
  for deg in 1..<degree:
    dA[deg] = xij * (A[i, deg] - psj * dA[deg-1])


proc update(psj: float64, X: CSCDataset, y: seq[float64], yPred: seq[float64],
            beta: float64, degree, j: int, loss: LossFunction,
            A: var Matrix, dA: var Vector): float64 {.inline.} =
  result = beta * psj
  var invStepSize: float64 = 0.0
  for (i, val) in X.getCol(j):
    computeDerivative(dA, A, psj, val, i, degree)
    result += loss.dloss(yPred[i], y[i]) * dA[degree-1]
    invStepSize += dA[degree-1]^2

  invStepSize = invStepSize*loss.mu + beta
  result /= invStepSize


proc updateAug(psj: float64, y, yPred: seq[float64], beta: float64,
               degree, j: int, loss: LossFunction, A: var Matrix,
               dA: var Vector): float64 {.inline.} =
  result = beta * psj
  let nSamples = len(y)
  var invStepSize = 0.0
  for i in 0..<nSamples:
    computeDerivative(dA, A, psj, 1.0, i, degree)
    result += loss.dloss(yPred[i], y[i]) * dA[degree-1]
    invStepSize += dA[degree-1]^2

  invStepSize = invStepSize*loss.mu + beta
  result /= invStepSize


proc epoch(X: CSCDataset, y: seq[float64], yPred: var seq[float64],
           P: var Tensor, beta: float64, degree, order, nAugments: int,
           loss: LossFunction, A: var Matrix, dA: var Vector): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[1]
  let nSamples = X.nSamples
  for s in 0..<nComponents:
    # compute cache
    anova(X, P, A, degree, order, s, nAugments)
    for j in 0..<nFeatures:
      var psj = P[order, s, j]
      let update = update(psj, X, y, yPred, beta, degree, j, loss, A, dA)
      result += abs(update)
      # synchronize
      for (i, val) in X.getCol(j):
        dA[0] = val
        for deg in 1..<degree:
          dA[deg] = val * (A[i, deg] - psj * dA[deg-1])
          A[i, deg] -= update * dA[deg-1]
        A[i, degree] -= update * dA[degree-1]
        yPred[i] -= update * dA[degree-1]
      P[order, s, j] -= update

    # for augmented features
    for j in nFeatures..<(nFeatures+nAugments):
      var psj = P[order, s, j]
      let update = updateAug(psj, y, yPred, beta, degree, j, loss, A, dA)
      result += abs(update)
      # synchronize
      for i in 0..<nSamples:
        dA[0] = 1.0
        for deg in 1..<degree:
          dA[deg] = A[i, deg] - P[order, s, j]*dA[deg-1]
          A[i, deg] -= update * dA[deg-1]
        A[i, degree] -= update * dA[degree-1]
        yPred[i] -= update * dA[degree-1]
      P[order, s, j] -= update


# optimized for degree=2, faster than above epoch proc
proc epochDeg2(X: CSCDataset, y: seq[float64], yPred: var seq[float64],
               P: var Tensor, beta: float64, order, nAugments: int,
               loss: LossFunction, cacheDeg2: var Vector): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[1]
  let nSamples = X.nSamples
  for s in 0..<nComponents:
    # compute cache. cacheDeg2[i] = \langle p_{s}, x_i \rangle
    for i in 0..<nSamples:
      cacheDeg2[i] = 0
      if nAugments == 1: cacheDeg2[i] = P[order, s, nFeatures]
    for j in 0..<nFeatures:
      for (i, val) in X.getCol(j):
        cacheDeg2[i] += val * P[order, s, j]

    for j in 0..<nFeatures:
      var psj = P[order, s, j]
      var update = beta * psj
      var invStepSize = 0.0
      for (i, val) in X.getCol(j):
        let dA = (cacheDeg2[i] - psj * val) * val
        update += loss.dloss(yPred[i], y[i]) * dA
        invStepSize += dA^2
      invStepSize = invStepSize*loss.mu + beta
      if invStepSize < 1e-12: 
        continue
      update /= invStepSize
      result += abs(update)
      # synchronize
      for (i, val) in X.getCol(j):
        yPred[i] -= update * (cacheDeg2[i] - psj * val) * val
        cacheDeg2[i] -= update * val
      P[order, s, j] -= update

    # for augmented features
    if nAugments == 1:
      var psj = P[order, s, nFeatures]
      var update = beta * psj
      var invStepSize = 0.0
      for i in 0..<nSamples:
        let dA = (cacheDeg2[i] - psj)
        update += loss.dloss(yPred[i], y[i]) * dA
        invStepSize += dA^2
      invStepSize = invStepSize*loss.mu + beta
      update /= invStepSize
      result += abs(update)
      # synchronize
      for i in 0..<nSamples:
        yPred[i] -= update * (cacheDeg2[i] - psj)
        cacheDeg2[i] -= update
      P[order, s, nFeatures] -= update


proc fit*(self: CoordinateDescent, X: CSCDataset, y: seq[float64],
          fm: var FactorizationMachine) =
  fm.init(X)
  let y = fm.checkTarget(y)
  var it: int
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    alpha0 = fm.alpha0 * float(nSamples)
    alpha = fm.alpha * float(nSamples)
    beta = fm.beta * float(nSamples)
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nAugments = fm.nAugments
    loss = newLossFunction(fm.loss)

  # caches
  var
    yPred = newSeqWith(nSamples, 0.0)
    A: Matrix = zeros([nSamples, degree+1])
    dA: Vector = zeros([degree])
    cacheDeg2: Vector = zeros([nSamples])
    colNormSq: Vector = zeros([nFeatures])
  
  # init caches
  for i in 0..<nSamples:
    A[i, 0] = 1.0
  if fitLinear:
   for j in 0..<nFeatures:
     for (_, val) in X.getCol(j):
       colNormSq[j] += val^2
  # compute prediction
  linear(X, fm.w, yPred)
  for i in 0..<nSamples:
    yPred[i] += fm.intercept

  for order in 0..<nOrders:
    for s in 0..<nComponents:
      anova(X, fm.P, A, degree-order, order, s, nAugments)
      for i in 0..<nSamples:
        yPred[i] += A[i, degree-order]
  
  for it in 0..<self.maxIter:
    var viol = 0.0

    if fitIntercept:
      viol += fitInterceptCD(fm.intercept, y, yPred, nSamples, alpha0, loss)

    if fitLinear:
      viol += fitLinearCD(fm.w, X, y, yPred, colNormSq, alpha, loss)
    
    for order in 0..<nOrders:
      if (degree-order) > 2:
        viol += epoch(X, y, yPred, fm.P, beta, degree-order, order, nAugments,
                      loss, A, dA)
      else:
        viol += epochDeg2(X, y, yPred, fm.P, beta, order, nAugments, loss,
                          cacheDeg2)

    if self.verbose:
      var meanLoss = 0.0
      for i in 0..<nSamples:
        meanLoss += loss.loss(yPred[i], y[i])
      meanLoss /= float(nSamples)
      stdout.write(fmt"Iteration: {align($(it+1), len($self.maxIter))}")
      stdout.write(fmt"   Violation: {viol:1.4e}")
      stdout.write(fmt"   Loss: {meanloss:1.4e}")
      stdout.write("\n")
      stdout.flushFile()

    if viol < self.tol:
      if self.verbose: echo("Converged at iteration ", it, ".")
      break
  if it == self.maxIter and self.verbose:
    echo("Objective did not converge. Increase maxIter.")
