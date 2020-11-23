import nimfm/tensor/tensor, nimfm/optimizer/optimizer_base, nimfm/loss
import sequtils, math
import ../model/fm_slow, fit_linear_slow
from ../regularizer/regularizers import newSquaredL12Slow
from cd_slow import computeDerivatives, computeInvStepSize, computeGrad, predict


type
  PBCDSlow*[L, R] = ref object of BaseCSCOptimizer
    ## Coordinate descent solver for test.
    gamma: float64
    loss: L
    reg: R
    sigma: float64
    rho: float64
    maxSearch: int
    shrink: bool
    shuffle: bool


proc newPBCDSlow*[L, R](maxIter = 100, alpha0=1e-6, alpha=1e-3, beta=1e-4,
                       gamma=1e-4, loss: L = newSquared(),
                       reg: R = newSquaredL12Slow(), sigma=0.01, rho=0.5,
                       maxSearch=0, shrink=false, shuffle=false,
                       verbose = 1, tol = 1e-3): PBCDSlow[L, R] =
  result = PBCDSlow[L, R](maxIter: maxIter, alpha0: alpha0, alpha: alpha, 
                          beta: beta, gamma: gamma, loss: loss, reg: reg,
                          tol: tol, verbose: verbose, sigma: sigma, rho: rho,
                          maxSearch: maxSearch, shrink: shrink,
                          shuffle: shuffle)


proc lineSearch[L, R](self: PBCDSlow[L, R], X: Matrix, y: seq[float64],
                      yPred: var Vector, P: var Tensor, j, order, degree: int,
                      w: Vector, intercept: float64,
                      grad, delta, old_p: var Vector,
                      oldLossVal, oldRegVal: float64) =
  var it  = 0
  var alpha = 1.0
  var newLossVal = 0.0
  var newRegVal = 0.0

  for i in 0..<X.shape[0]:
    newLossVal += self.loss.loss(y[i], yPred[i])
  newLossVal /= float(X.shape[0])
  newRegVal = self.gamma * self.reg.eval(P[order].T, degree-order)

  newRegVal += 0.5 * self.beta * norm(P[order], 2)^2 
  var cond = - dot(grad, delta) + newRegVal - oldRegVal
  var decreasing = newLossVal + newRegVal - oldLossVal - oldRegVal

  while not (decreasing <= self.sigma * alpha * cond):
    if it >= self.maxSearch: break
    # update!
    alpha *= self.rho
    for s in 0..<P.shape[1]:
      P[order, s, j] = old_p[s] - alpha * delta[s]
    # compute loss
    predict(P, w, intercept, X, yPred, degree)
    newLossVal = 0.0
    for i in 0..<X.shape[0]:
      newLossVal += self.loss.loss(y[i], yPred[i])
    newLossVal /= float(X.shape[0])
    # compute regularization
    newRegVal = self.gamma * self.reg.eval(P[order].T, degree-order)
    decreasing = newLossVal + newRegVal - oldLossVal - oldLossVal
    decreasing += 0.5 * self.beta * (norm(P[order], 2)^2 - norm(old_p, 2)^2)

    inc(it)
  delta *= alpha


proc epoch[L, R](self: PBCDSlow[L, R], X: Matrix, y: seq[float64], yPred: var seq[float64],
                 P: var Tensor, degree, order, nAugments: int,
                 dA: var Vector, w: Vector, intercept: float64, 
                 grad, invStepSizes, delta, old_p: var Vector): float64 =
  result = 0.0
  let nFeatures = X.shape[1]
  let nComponents = P.shape[1]
  let nSamples = float(X.shape[0])
  let beta = self.beta * nSamples
  var invStepSize = 0.0
  var lossVal = 0.0
  var regVal = 0.0
  var nnz: int = 0

  for j in 0..<nFeatures+nAugments:
    predict(P, w, intercept, X, yPred, degree)
    # compute gradient and invStepSize
    for s in 0..<nComponents:
      computeDerivatives(P, X, dA, degree-order, order, s, j, nAugments)
      invStepSizes[s] = computeInvStepSize(0.0, self.loss, dA)
      grad[s] = computeGrad(P, y, yPred, beta, order, s, j, self.loss, dA)
    grad /= nSamples
    invStepSizes /= nSamples
    
    # compute invStepSize
    if self.maxSearch != 0: 
      invStepSize = max(invStepSizes)
      lossVal = 0.0
      regVal = 0.0
      for i in 0..<X.shape[0]:
        lossVal += self.loss.loss(y[i], yPred[i])
      lossVal /= nSamples
      regVal = self.gamma * self.reg.eval(P[order].T, degree-order)
      regVal += 0.5 * self.beta * norm(P[order], 2)^2
    else:
      invStepSize = sum(invStepSizes)
    invStepSize += self.beta
    invStepSize = max(invStepSize, 1e-12)

    # gradient update and proximal update
    for s in 0..<nComponents:
      old_p[s] = P[order, s, j]
      P[order, s, j] -= grad[s] / invStepSize
    self.reg.prox(P[order], self.gamma / invStepSize, degree-order, j)

    predict(P, w, intercept, X, yPred, degree)
    # line search?
    for s in 0..<nComponents:
      nnz = 0
      if P[order, s, j] != 0.0:
        inc(nnz)
    if not (self.maxSearch == 0):
      for s in 0..<nComponents:
        delta[s] = - P[order, s, j] + old_p[s]
      lineSearch(self, X, y, yPred, P, j, order, degree, w, intercept,
                 grad, delta, old_p, lossVal, regVal)


proc fit*[L, R](self: PBCDSlow[L, R], X: Matrix, y: seq[float64],
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
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nAugments = fm.nAugments
    loss = self.loss
  var
    grad = zeros([fm.P.shape[1]])
    invStepSizes = zeros([fm.P.shape[1]])
    delta = zeros([fm.P.shape[1]])
    old_p = zeros([fm.P.shape[1]])

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
      viol += epoch(self, X, y, yPred, fm.P, degree, order, nAugments,
                    dA, fm.w, fm.intercept, grad, invStepSizes, delta, old_p)
