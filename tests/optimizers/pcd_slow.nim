import nimfm/tensor/tensor, nimfm/optimizers/optimizer_base, nimfm/loss
import sequtils, math
import ../models/fm_slow, fit_linear_slow
from ../regularizers/regularizers import newSquaredL12Slow
from cd_slow import computeDerivatives, computeInvStepSize, computeGrad, predict


type
  PCDSlow*[L, R] = ref object of BaseCSCOptimizer
    ## Coordinate descent solver for test.
    gamma: float64
    loss: L
    reg: R


proc newPCDSlow*[L, R](maxIter = 100, alpha0=1e-6, alpha=1e-3, beta=1e-4,
                       gamma=1e-4, loss: L = newSquared(),
                       reg: R = newSquaredL12Slow(),
                       verbose = 1, tol = 1e-3): PCDSlow[L, R] =
  result = PCDSlow[L, R](maxIter: maxIter, alpha0: alpha0, alpha: alpha, 
                         beta: beta, gamma: gamma, loss: loss, reg: reg,
                         tol: tol, verbose: verbose)


proc epoch[L, R](X: Matrix, y: seq[float64], yPred: var seq[float64],
                 P: var Tensor, beta: float64, degree, order, nAugments: int,
                 loss: L, dA: var Vector, w: Vector,
                 intercept: float64, gamma: float64, reg: R): float64 =
  result = 0.0
  let nFeatures = X.shape[1]
  let nComponents = P.shape[1]
  for s in 0..<nComponents:
    for j in 0..<nFeatures+nAugments:
      computeDerivatives(P, X, dA, degree-order, order, s, j, nAugments)
      let invStepSize = computeInvStepSize(beta, loss, dA)
      if invStepSize < 1e-12: continue
      let update = computeGrad(P, y, yPred, beta, order, s, j, loss, dA) / invStepSize
      P[order, s, j] -= update
      reg.prox(P[order], gamma / invStepSize, degree-order, s, j)
      result += abs(update)
      # naive synchronize
      predict(P, w, intercept, X, yPred, degree)


proc fit*[L, R](self: PCDSlow[L, R], X: Matrix, y: seq[float64],
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
                    loss, dA, fm.w, fm.intercept, self.gamma*float(nSamples),
                    self.reg)
