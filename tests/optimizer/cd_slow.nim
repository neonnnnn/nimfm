import nimfm/tensor/tensor, nimfm/optimizer/optimizer_base
import sequtils, math
import ../model/fm_slow, ../kernels_slow, fit_linear_slow, ../comb
from nimfm/loss import newSquared

type
  CDSlow*[L] = ref object of BaseCSCOptimizer
    ## Coordinate descent solver for test.
    loss: L

proc newCDSlow*[L](maxIter = 100, alpha0 = 1e-6, alpha = 1e-3, beta = 1e-3,
                   loss: L =newSquared(), verbose = 1,
                   tol = 1e-3): CDSlow[L] =
  result = CDSlow[L](maxIter: maxIter, alpha0: alpha0, alpha: alpha,
                     beta: beta, loss: loss, tol: tol, verbose: verbose)


proc predict*(P: Tensor, w: Vector, intercept: float64, X: Matrix,
              yPred: var seq[float64], degree: int) =

  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    nAugments = P.shape[2] - nFeatures
    nOrders = P.shape[0]
    nComponents = P.shape[1]
  for i in 0..<nSamples:
    yPred[i] = intercept

  for j in 0..<nFeatures:
    for i in 0..<nSamples:
      yPred[i] += w[j] * X[i, j]

  for order in 0..<nOrders:
    for s in 0..<nComponents:
      for i in 0..<nSamples:
        let anova = anovaSlow(X, P[order], i, degree-order,
                              s, nFeatures, nAugments)
        yPred[i] += anova


# compute ient naively
# dA/dpj =  anova_without_j  * xj
proc computeDerivatives*(P: Tensor, X: Matrix, dA: var Vector,
                         degree, order, s, j, nAugments: int) =
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
 
  for i in 0..<nSamples:
    dA[i] = 0
  for i in 0..<nSamples:
    for indices in combNotj(nFeatures+nAugments, degree-1, j):
      var prod = 1.0
      for j2 in indices:
        prod *= P[order, s, j2]
        if j2 < nFeatures:
          prod *= X[i, j2]
      dA[i] += prod
    if j < nFeatures:
      dA[i] *= X[i, j]


proc computeGrad*[L](P: Tensor, y, yPred: seq[float64], beta: float64,
                     order, s, j: int, loss: L, dA: var Vector): float64 =
  result = beta * P[order, s, j]
  let nSamples = dA.shape[0]
  for i in 0..<nSamples:
    result += loss.dloss(y[i], yPred[i]) * dA[i]


proc computeInvStepSize*[L](beta: float64, loss: L, dA: var Vector): float64 =
  let nSamples = dA.shape[0]
  result = 0.0
  for i in 0..<nSamples:
    result += dA[i]^2
  result = result*loss.mu + beta


proc epoch[L](X: Matrix, y: seq[float64], yPred: var seq[float64],
              P: var Tensor, beta: float64, degree, order, nAugments: int,
              loss: L, dA: var Vector, w: Vector,
              intercept: float64): float64 =
  result = 0.0
  let nFeatures = X.shape[1]
  let nComponents = P.shape[1]
  for s in 0..<nComponents:
    for j in 0..<nFeatures+nAugments:
      computeDerivatives(P, X, dA, degree-order, order, s, j, nAugments)
      let invStepSize = computeInvStepSize(beta, loss, dA)
      let update = computeGrad(P, y, yPred, beta, order, s, j, loss, dA) / invStepSize
      P[order, s, j] -= update
      result += abs(update)
      # naive synchronize
      predict(P, w, intercept, X, yPred, degree)


proc fit*[L](self: CDSlow[L], X: Matrix, y: seq[float64],
             fm: var FMSlow) =
  fm.init(X)
  let y = fm.checkTarget(y)
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    alpha0 = self.alpha0 * float(nSamples)
    alpha = self.alpha * float(nSamples)
    beta = self.beta * float(nSamples)
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nAugments = fm.nAugments
    loss = self.loss

  # caches
  var
    yPred = newSeqWith(nSamples, fm.intercept)
    dA: Vector = zeros([nSamples])
    colNormSq: Vector = zeros([nFeatures])

  if fitLinear:
    for j in 0..<nFeatures:
      for i in 0..<nSamples:
        colNormSq[j] += X[i, j]^2

  predict(fm.P, fm.w, fm.intercept, X, yPred, degree)
  for it in 0..<self.maxIter:
    var viol = 0.0
    if fitIntercept:
      viol += fitInterceptCD(fm.intercept, y, yPred, nSamples, alpha0, loss)
      predict(fm.P, fm.w, fm.intercept, X, yPred, degree)

    if fitLinear:
      viol += fitLinearCD(fm.w, X, y, yPred, colNormSq, alpha, loss)
      predict(fm.P, fm.w, fm.intercept, X, yPred, degree)
    
    for order in 0..<nOrders:
      viol += epoch(X, y, yPred, fm.P, beta, degree, order, nAugments,
                    loss, dA, fm.w, fm.intercept)
