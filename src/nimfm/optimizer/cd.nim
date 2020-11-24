import ../dataset, ../tensor/tensor, ../kernels, ../extmath, ../loss
import ../model/fm_base, ../model/factorization_machine
import fit_linear, optimizer_base, utils
import sequtils, math, sugar

type
  CD*[L] = ref object of BaseCSCOptimizer
    ## Coordinate descent solver.
    loss*: L


proc newCD*[L](maxIter = 100, alpha0=1e-6, alpha = 1e-3, beta = 1e-3,
               loss: L = newSquared(), verbose = 1, tol = 1e-3): CD[L] =
  ## Creates new CD.
  ## maxIter: Maximum number of iteration. At each iteration,
  ##          all parameters are updated once by using all samples.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  result = CD[L](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta,
    loss: loss, tol: tol, verbose: verbose)


proc computeDerivative*(dA: var Vector, A: var Matrix, psj, xij: float64,
                        i, degree: int) {.inline.} =
  dA[0] = xij
  for deg in 1..<degree:
    dA[deg] = xij * (A[i, deg] - psj * dA[deg-1])


proc update*[L](psj: float64, X: ColDataset, y, yPred: seq[float64],
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


proc epoch[L](X: ColDataset, y: seq[float64], yPred: var seq[float64],
              P: var Matrix, beta: float64, degree: int,
              loss: L, A: var Matrix, dA: var Vector): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[0]
  for s in 0..<nComponents:
    # compute cache
    anova(X, P, A, degree, s)
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

    
# optimized for degree=2, faster than above epoch proc
proc epochDeg2[L](X: ColDataset, y: seq[float64], yPred: var seq[float64],
                  P: var Matrix, beta: float64, 
                  loss: L, cacheDeg2: var Vector): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  let nComponents = P.shape[0]
  for s in 0..<nComponents:
    # compute cache. cacheDeg2[i] = \langle p_{s}, x_i \rangle
    cacheDeg2[0..^1] = 0
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


proc fit*[L](self: CD[L], X: ColDataset, y: seq[float64],
             fm: FactorizationMachine,
             callback: (CD[L], FactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by coordinate descent.
  fm.init(X)

  let y = fm.checkTarget(y)
  let
    nSamples = X.nSamples
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    nAugments = fm.nAugments
    alpha0 = self.alpha0 * float(nSamples)
    alpha = self.alpha * float(nSamples)
    beta = self.beta * float(nSamples)
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept

  # caches
  var
    yPred = newSeqWith(nSamples, 0.0)
    A: Matrix = zeros([nSamples, degree+1])
    dA: Vector = zeros([degree])
    cacheDeg2: Vector = zeros([nSamples])
    colNormSq: Vector
    isConverged = false
  
  # init caches
  A[0..^1, 0] = 1.0
  if fitLinear:
    colNormSq = norm(X, p=2, axis=0)
    colNormSq *= colNormSq
  # compute prediction
  linear(X, fm.w, yPred)
  yPred += fm.intercept

  X.addDummyFeature(1.0, fm.nAugments)
  for order in 0..<nOrders:
    for s in 0..<nComponents:
      anova(X, fm.P[order], A, degree-order, s)
      yPred += A[0..^1, degree-order]
  
  if self.verbose > 0: # echo header
    echoHeader(self.maxIter)

  for it in 0..<self.maxIter:
    var viol = 0.0

    X.removeDummyFeature(nAugments) # remove dummy for linear term
    if fitIntercept:
      viol += fitInterceptCD(fm.intercept, y, yPred, nSamples, alpha0, self.loss)

    if fitLinear:
      viol += fitLinearCD(fm.w, X, y, yPred, colNormSq, alpha, self.loss)
    
    X.addDummyFeature(1.0, nAugments)
    for order in 0..<nOrders:
      if (degree-order) > 2:
        viol += epoch(X, y, yPred, fm.P[order], beta, degree-order, self.loss,
                      A, dA)
      else:
        viol += epochDeg2(X, y, yPred, fm.P[order], beta, self.loss, cacheDeg2)

    if not callback.isNil:
      callback(self, fm)

    if self.verbose > 0:
      var lossVal = 0.0
      for i in 0..<nSamples:
        lossVal += self.loss.loss(y[i], yPred[i])
      lossVal /= float(nSamples)
      let reg = regularization(fm.P, fm.w, fm.intercept, 
                               alpha0, alpha, beta) / float(nSamples)
      echoInfo(it+1, self.maxIter, viol, lossVal, reg)

    if viol < self.tol:
      if self.verbose > 0: echo("Converged at iteration ", it+1, ".")
      isConverged = true
      break

  X.removeDummyFeature(fm.nAugments)

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")
