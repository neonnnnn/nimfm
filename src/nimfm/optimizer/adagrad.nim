import ../dataset, ../tensor/tensor, ../model/factorization_machine, ../model/params
import ../loss
from ../model/fm_base import checkTarget
import fit_linear, optimizer_base, utils
from sgd import predictWithGrad, stoppingCriterion, transpose
import sequtils, math, random, sugar

type
  AdaGrad*[L] = ref object of BaseCSROptimizer
    loss*: L
    eta0*: float64
    eps: float64
    shuffle*: bool
    it*: int
    g_sum*: Params
    g_norm*: Params
    nCalls*: int


proc newAdaGrad*[L](maxIter = 100, eta0=0.1, alpha0=1e-6, alpha=1e-3,
                    beta=1e-3, loss: L=newSquared(), eps = 1e-10, verbose = 1,
                    tol = 1e-3, shuffle = true, nCalls = -1): AdaGrad[L] =
  ## Creates new AdaGrad.
  ## maxIter: Maximum number of iteration. At each epoch,
  ##          all parameters are updated nSamples times.
  ## eta0: Step-size hyperparameter.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## eps: A non-negative float to add the denominator in order to
  ##      improve numerical stability.
  ## verbose: Whether to print information on optimization processes or not.
  ## tol: Tolerance hyperparameter for stopping criterion.
  ## shuffle: How to choose one instance: cyclic (false) or random permutation
  ##          (true).
  ## nCalls: Frequency with which callback must be called in the inner loop.
  ##         If nCalls <= 0 or multi-thread case, callback is called per one epoch.

  result = AdaGrad[L](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, loss: loss,
    eta0: eta0, eps: eps, tol: tol, it: 1, verbose: verbose, shuffle: shuffle,
    nCalls: nCalls)


proc init*[L](self: AdaGrad[L], P: Tensor, w: Vector,
              warmStart, fitLinear, fitIntercept: bool) =
  if not warmStart:
    self.it = 1

  if self.it == 1:
    self.g_sum = newParams(P.shape, w.len, fitLinear, fitIntercept)
    self.g_norm = newParams(P.shape, w.len, fitLinear, fitIntercept)
    self.g_norm <- self.eps
  else:
    if P.shape != self.g_sum.P.shape:
      let msg = "warmStart=true but P.shape != g_sum.P.shape."
      raise newException(ValueError, msg)

  if self.verbose > 0: # echo header
    echoHeader(self.maxIter)


proc finalize*[L](self: AdaGrad[L], P: var Tensor, w: var Vector,
                  intercept: var float64, fitLinear, fitIntercept: bool) =
  let it = float(self.it-1)
  let denom = self.eta0*it*self.beta
  for order in 0..<P.shape[0]:
    for j in 0..<P.shape[1]:
      for s in 0..<P.shape[2]:
        P[order, j, s] = -self.eta0 * self.g_sum.P[order, j, s]
        P[order, j, s] /= denom + sqrt(self.g_norm.P[order, j, s])

  # for w and intercept
  if fitIntercept:
    let denom = sqrt(self.g_norm.intercept) + self.eta0*it*self.alpha0
    intercept = -self.eta0 * self.g_sum.intercept / denom

  if fitLinear:
    let denom = self.eta0 * it * self.alpha 
    for j in 0..<len(w):
      w[j] = -self.eta0 * self.g_sum.w[j]
      w[j] /= denom + sqrt(self.g_norm.w[j])


proc update*[L](self: AdaGrad[L], P: var Tensor, w: var Vector, 
                intercept: var float64, X: RowDataset, i: int,
                nAugments: int, fitLinear, fitIntercept: bool): float64 =
  let it = float(self.it-1)
  let tmp = self.eta0 * it * self.beta
  X.addDummyFeature(1.0, nAugments)
  for order in 0..<P.shape[0]:
    for j in X.getRowIndices(i):
      for s in 0..<P.shape[2]:
        let pjs = P[order, j, s]
        let denom = tmp + sqrt(self.g_norm.P[order, j, s])
        P[order, j, s] = - (self.eta0 * self.g_sum.P[order, j, s]) / denom
        result += abs(pjs-P[order, j, s])
  X.removeDummyFeature(nAugments)
  # for w and intercept
  if fitIntercept:
    let old = intercept
    let denom = sqrt(self.g_norm.intercept) + self.eta0*it*self.alpha0
    intercept = -self.eta0 * self.g_sum.intercept / denom
    result += abs(old - intercept)

  if fitLinear:
    result += fitLinearAdaGrad(w, self.g_sum.w, self.g_norm.w, X, i, self.alpha,
                               self.eta0, it)


proc updateG*[L](self: AdaGrad[L], X: RowDataset, df: Tensor, i: int,
                 yi, yPred: float64, nAugments: int,
                 fitLinear, fitIntercept: bool) =
  X.addDummyFeature(1.0, nAugments)
  # update g_sum and g_norms
  let dL = self.loss.dloss(yi, yPred)
  for order in 0..<self.g_sum.P.shape[0]:
    for j in X.getRowIndices(i):
      for s in 0..<self.g_sum.P.shape[2]:
        let grad = dL*df[order, j, s]
        self.g_sum.P[order, j, s] += grad 
        self.g_norm.P[order, j, s] += grad*grad
  X.removeDummyFeature(nAugments)

  if fitIntercept:
    self.g_sum.intercept += dL
    self.g_norm.intercept += dL^2

  if fitLinear:
    for (j, val) in X.getRow(i):
      self.g_sum.w[j] += dL * val
      self.g_norm.w[j] += (dL*val)^2
 

proc fit*[L](self: AdaGrad[L], X: RowDataset, y: seq[float64],
             fm: FactorizationMachine,
             callback: (AdaGrad[L], FactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  fm.init(X)

  let y = fm.checkTarget(y)
  let
    nSamples = X.nSamples
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    nAugments = fm.nAugments
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
  var
    A: Matrix = zeros([nComponents, degree+1])
    indices = toSeq(0..<nSamples)
    P: Tensor = zeros([nOrders, fm.P.shape[2], nComponents])
    dA: Tensor = zeros(P.shape)
    isConverged = false
  
  # initialization
  init(self, P, fm.w, fm.warmStart, fitLinear, fitIntercept)
  # copy for fast training
  transpose(P, fm.P)

  for epoch in 0..<self.maxIter:
    var viol = 0.0
    var runningLoss = 0.0
    if X.nCached == X.nSamples and self.shuffle: shuffle(indices)
    
    for i in indices:
      # update parameters lazily
      if self.it != 1:
        viol += update(self, P, fm.w, fm.intercept, X, i, nAugments,
                       fitLinear, fitIntercept)
      let yPred = predictWithGrad(X, i, P, fm.w, fm.intercept, A,
                                  dA, degree, nAugments)
      runningLoss += self.loss.loss(y[i], yPred)
      updateG(self, X, dA, i, y[i], yPred, nAugments, fitLinear, fitIntercept)

      if self.nCalls > 0 and self.it mod self.nCalls == 0:
        if not callback.isNil:
          finalize(self, P, fm.w, fm.intercept, fitLinear, fitIntercept)
          transpose(fm.P, P)
          callback(self, fm)
      inc(self.it)

    # one epoch done
    runningLoss /= float(nSamples)
    if not callback.isNil:
      finalize(self, P, fm.w, fm.intercept, fitLinear, fitIntercept)
      transpose(fm.P, P)
      callback(self, fm)
    
    let isContinue = stoppingCriterion(
      P, fm.w, fm.intercept, self.alpha0, self.alpha, self.beta, runningLoss,
      viol, self.tol, self.verbose, epoch, self.maxIter, isConverged)
    if not isContinue: break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")
  
  # finalize
  finalize(self, P, fm.w, fm.intercept, fitLinear, fitIntercept)
  transpose(fm.P, P)