import nimfm/loss, nimfm/dataset, nimfm/tensor, nimfm/kernels,
    nimfm/factorization_machine
import nimfm/optimizers/fit_linear, nimfm/optimizers/base
import sequtils, math


type
  CoordinateDescentSlow* = ref object of BaseCSCOptimizer
    ## Coordinate descent solver for test.

proc newCoordinateDescentSlow*(maxIter = 100, verbose = true, tol = 1e-3):
                               CoordinateDescentSlow =
  result = CoordinateDescentSlow(maxIter: maxIter, tol: tol, verbose: verbose)


proc predict(P: Tensor, w: Vector, intercept: float64, X: CSCDataset,
             yPred: var seq[float64], A: var Matrix, degree: int) =

  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    nAugments = P.shape[2] - nFeatures
    nOrders = P.shape[0]
    nComponents = P.shape[1]
  for i in 0..<nSamples:
    yPred[i] = intercept

  for j in 0..<nFeatures:
    for (i, val) in X.getCol(j):
      yPred[i] += w[j] * val

  for order in 0..<nOrders:
    for s in 0..<nComponents:
      anova(X, P, A, degree-order, order, s, nAugments)
      for i in 0..<nSamples:
        yPred[i] += A[i, degree-order]


proc anovaWithoutOneElement(P: Tensor, X: CSCDataset, A: var Matrix,
                            degree, order, s, notj, nAugments: int) =
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    degree = degree-order-1
  for i in 0..<nSamples:
    A[i, 0] = 1
    for deg in 1..<(degree+1):
      A[i, deg] = 0
  for j in 0..<nFeatures:
    if j != notj:
      for (i, val) in X.getCol(j):
        for deg in 0..<degree:
          A[i, degree-deg] += A[i, degree-deg-1] * P[order, s, j] * val

  # for augmented features
  for j in nFeatures..<(nFeatures+nAugments):
    if j != notj:
      for i in 0..<nSamples:
        for deg in 0..<degree:
          A[i, degree-deg] += A[i, degree-deg-1] * P[order, s, j]


proc update(P: Tensor, X: CSCDataset, y: seq[float64], yPred: seq[float64],
            beta: float64, degree, order, s, j, nAugments: int,
            loss: LossFunction, A: var Matrix): float64 =
  result = beta * P[order, s, j]

  var invStepSize: float64 = 0.0
  anovaWithoutOneElement(P, X, A, degree, order, s, j, nAugments)
  for (i, val) in X.getCol(j):
    result += loss.dloss(yPred[i], y[i]) * A[i, degree-order-1] * val
    invStepSize += (A[i, degree-order-1]*val)^2

  invStepSize = invStepSize*loss.mu + beta
  result /= invStepSize


proc updateAug(P: Tensor, X: CSCDataset, y, yPred: seq[float64], beta: float64,
               degree, order, s, j, nAugments: int, loss: LossFunction,
               A: var Matrix): float64 {.inline.} =
  result = beta * P[order, s, j]
  let nSamples = X.nSamples
  var invStepSize = 0.0
  anovaWithoutOneElement(P, X, A, degree, order, s, j, nAugments)
  for i in 0..<nSamples:
    result += loss.dloss(yPred[i], y[i]) * A[i, degree-order-1]
    invStepSize += A[i, degree-order-1]^2

  invStepSize = invStepSize*loss.mu + beta
  result /= invStepSize


proc epoch(X: CSCDataset, y: seq[float64], yPred: var seq[float64],
           P: var Tensor, beta: float64, degree, order, nAugments: int,
           loss: LossFunction, A: var Matrix, w: Vector,
               intercept: float64): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[1]
  for s in 0..<nComponents:
    for j in 0..<nFeatures:
      let update = update(P, X, y, yPred, beta, degree, order, s, j, nAugments,
                          loss, A)
      P[order, s, j] -= update
      result += abs(update)
      # naive synchronize
      predict(P, w, intercept, X, yPred, A, degree)

    # for augmented features
    for j in nFeatures..<(nFeatures+nAugments):
      let update = updateAug(P, X, y, yPred, beta, degree, order, s, j,
                             nAugments, loss, A)
      result += abs(update)
      P[order, s, j] -= update
      # naive synchronize
      predict(P, w, intercept, X, yPred, A, degree)


proc fit*(self: CoordinateDescentSlow, X: CSCDataset, y: seq[float64],
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
    yPred = newSeqWith(nSamples, fm.intercept)
    A: Matrix = zeros([nSamples, degree+1])
    colNormSq: Vector = zeros([nFeatures])

  for i in 0..<nSamples:
    A[i, 0] = 1.0
    for deg in 1..<(degree+1):
      A[i, deg] = 0.0

  if fitLinear:
    for j in 0..<nFeatures:
      for (_, val) in X.getCol(j):
        colNormSq[j] += val^2

  predict(fm.P, fm.w, fm.intercept, X, yPred, A, degree)
  for it in 0..<self.maxIter:
    var viol = 0.0
    if fitIntercept:
      viol += fitInterceptCD(fm.intercept, y, yPred, nSamples, alpha0, loss)
      predict(fm.P, fm.w, fm.intercept, X, yPred, A, degree)
    if fitLinear:
      viol += fitLinearCD(fm.w, X, y, yPred, colNormSq, alpha, loss)
      predict(fm.P, fm.w, fm.intercept, X, yPred, A, degree)
    for order in 0..<nOrders:
      viol += epoch(X, y, yPred, fm.P, beta, degree, order, nAugments,
                    loss, A, fm.w, fm.intercept)
