import ../dataset, ../tensor, ../kernels, ../factorization_machine, ../extmath
import ../fm_base
import fit_linear, optimizer_base
import sequtils, math, strformat, strutils

type
  CD* = ref object of BaseCSCOptimizer
    ## Coordinate descent solver.


proc newCD*(maxIter = 100, verbose = 1, tol = 1e-3): CD =
  ## Creates new CD.
  ## maxIter: Maximum number of iteration. At each iteration,
  ##          all parameters are updated once by using all samples.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyper-parameter for stopping criterion.
  result = CD(maxIter: maxIter, tol: tol, verbose: verbose)


proc computeDerivative*(dA: var Vector, A: var Matrix, psj, xij: float64,
                        i, degree: int) {.inline.} =
  dA[0] = xij
  for deg in 1..<degree:
    dA[deg] = xij * (A[i, deg] - psj * dA[deg-1])


proc update*[L](psj: float64, X: CSCDataset, y, yPred: seq[float64],
             beta: float64, degree, j: int, loss: L, A: var Matrix,
             dA: var Vector): tuple[update, invStepSize: float64] {.inline.} =
  result[0] = beta * psj
  result[1] = 0.0
  for (i, val) in X.getCol(j):
    computeDerivative(dA, A, psj, val, i, degree)
    result[0] += loss.dloss(y[i], yPred[i]) * dA[degree-1]
    result[1] += dA[degree-1]^2

  result[1] *= loss.mu
  result[1] += beta


proc updateAug*[L](psj: float64, y, yPred: seq[float64], beta: float64,
                   degree, j: int, loss: L, A: var Matrix,
                   dA: var Vector): tuple[update, invStepSize: float64] =
  result[0] = beta * psj
  result[1] = 0.0
  let nSamples = len(y)
  for i in 0..<nSamples:
    computeDerivative(dA, A, psj, 1.0, i, degree)
    result[0] += loss.dloss(y[i], yPred[i]) * dA[degree-1]
    result[1] += dA[degree-1]^2

  result[1] *= loss.mu
  result[1] += beta


proc epoch[L](X: CSCDataset, y: seq[float64], yPred: var seq[float64],
              P: var Matrix, beta: float64, degree, nAugments: int,
              loss: L, A: var Matrix, dA: var Vector): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[0]
  let nSamples = X.nSamples
  for s in 0..<nComponents:
    # compute cache
    anova(X, P, A, degree, s, nAugments)
    for j in 0..<nFeatures:
      let psj = P[s, j]
      var (update, invStepSize) = update(psj, X, y, yPred, beta, degree, 
                                         j, loss, A, dA)
      update /= invStepSize
      P[s, j] -= update
      result += abs(update)
      # synchronize
      for (i, val) in X.getCol(j):
        dA[0] = val
        for deg in 1..<degree:
          dA[deg] = val * (A[i, deg] - psj * dA[deg-1])
          A[i, deg] -= update * dA[deg-1]
        A[i, degree] -= update * dA[degree-1]
        yPred[i] -= update * dA[degree-1]

    # for augmented features
    for j in nFeatures..<(nFeatures+nAugments):
      let psj = P[s, j]
      var (update, invStepSize) = updateAug(psj, y, yPred, beta, degree, j, 
                                            loss, A, dA)
      update /= invStepSize
      P[s, j] -= update
      result += abs(update)
      # synchronize
      for i in 0..<nSamples:
        dA[0] = 1.0
        for deg in 1..<degree:
          dA[deg] = A[i, deg] - psj*dA[deg-1]
          A[i, deg] -= update * dA[deg-1]
        A[i, degree] -= update * dA[degree-1]
        yPred[i] -= update * dA[degree-1]


# optimized for degree=2, faster than above epoch proc
proc epochDeg2[L](X: CSCDataset, y: seq[float64], yPred: var seq[float64],
                  P: var Matrix, beta: float64, nAugments: int,
                  loss: L, cacheDeg2: var Vector): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[0]
  let nSamples = X.nSamples
  for s in 0..<nComponents:
    # compute cache. cacheDeg2[i] = \langle p_{s}, x_i \rangle
    for i in 0..<nSamples:
      cacheDeg2[i] = 0
      if nAugments == 1: cacheDeg2[i] = P[s, nFeatures]
    for j in 0..<nFeatures:
      for (i, val) in X.getCol(j):
        cacheDeg2[i] += val * P[s, j]

    for j in 0..<nFeatures:
      var psj = P[s, j]
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
      result += abs(update)
      # synchronize
      for (i, val) in X.getCol(j):
        yPred[i] -= update * (cacheDeg2[i] - psj * val) * val
        cacheDeg2[i] -= update * val
      P[s, j] -= update

    # for augmented features
    if nAugments == 1:
      var psj = P[s, nFeatures]
      var update = beta * psj
      var invStepSize = 0.0
      for i in 0..<nSamples:
        let dA = (cacheDeg2[i] - psj)
        update += loss.dloss(y[i], yPred[i]) * dA
        invStepSize += dA^2
      invStepSize = invStepSize*loss.mu + beta
      update /= invStepSize
      P[s, nFeatures] -= update
      result += abs(update)
      # synchronize
      for i in 0..<nSamples:
        yPred[i] -= update * (cacheDeg2[i] - psj)
        cacheDeg2[i] -= update


proc fit*[L](self: CD, X: CSCDataset, y: seq[float64],
             fm: var FactorizationMachine[L]) =
  ## Fits the factorization machine on X and y by coordinate descent.
  fm.init(X)
  let y = fm.checkTarget(y)
  let
    nSamples = X.nSamples
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    alpha0 = fm.alpha0 * float(nSamples)
    alpha = fm.alpha * float(nSamples)
    beta = fm.beta * float(nSamples)
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nAugments = fm.nAugments

  # caches
  var
    yPred = newSeqWith(nSamples, 0.0)
    A: Matrix = zeros([nSamples, degree+1])
    dA: Vector = zeros([degree])
    cacheDeg2: Vector = zeros([nSamples])
    colNormSq: Vector
    isConverged = false
  
  # init caches
  for i in 0..<nSamples:
    A[i, 0] = 1.0
  if fitLinear:
    colNormSq = norm(X, p=2, axis=0)
    colNormSq *= colNormSq
 # compute prediction
  linear(X, fm.w, yPred)
  for i in 0..<nSamples:
    yPred[i] += fm.intercept

  for order in 0..<nOrders:
    for s in 0..<nComponents:
      anova(X, fm.P[order], A, degree-order, s, nAugments)
      for i in 0..<nSamples:
        yPred[i] += A[i, degree-order]
  
  for it in 0..<self.maxIter:
    var viol = 0.0

    if fitIntercept:
      viol += fitInterceptCD(fm.intercept, y, yPred, nSamples, alpha0, fm.loss)

    if fitLinear:
      viol += fitLinearCD(fm.w, X, y, yPred, colNormSq, alpha, fm.loss)
    
    for order in 0..<nOrders:
      if (degree-order) > 2:
        viol += epoch(X, y, yPred, fm.P[order], beta, degree-order, nAugments,
                      fm.loss, A, dA)
      else:
        viol += epochDeg2(X, y, yPred, fm.P[order], beta, nAugments, fm.loss,
                          cacheDeg2)

    if self.verbose > 0:
      var meanLoss = 0.0
      for i in 0..<nSamples:
        meanLoss += fm.loss.loss(y[i], yPred[i])
      meanLoss /= float(nSamples)
      let epochAligned = align($(it+1), len($self.maxIter))
      stdout.write(fmt"Epoch: {epochAligned}")
      stdout.write(fmt"   Violation: {viol:1.4e}")
      stdout.write(fmt"   Loss: {meanloss:1.4e}")
      stdout.write("\n")
      stdout.flushFile()

    if viol < self.tol:
      if self.verbose > 0: echo("Converged at iteration ", it+1, ".")
      isConverged = true
      break
  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")
