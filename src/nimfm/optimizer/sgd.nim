import ../dataset, ../tensor/tensor, ../model/factorization_machine, ../loss
from ../model/fm_base import checkTarget
import fit_linear, optimizer_base, utils
import sequtils, math, random, sugar, strformat

type
  SchedulingKind* = enum
    constant = "constant",
    optimal = "optimal",
    invscaling = "invscaling",
    pegasos = "pegasos"

  SGD*[L] = ref object of BaseCSROptimizer
    loss*: L
    eta0: float64
    scheduling: SchedulingKind
    power: float64
    it*: int
    shuffle*: bool
    nCalls*: int


proc newSGD*[L](maxIter=100, eta0 = 0.01, alpha0=1e-6, alpha=1e-3, beta=1e-3,
                loss: L = newSquared(), scheduling = optimal, power = 1.0,
                verbose = 1, tol = 1e-3, shuffle = true, nCalls = -1): SGD[L] =
  ## Creates new SGD.
  ## maxIter: Maximum number of iteration. At each iteration,
  ##          all parameters are updated nSamples times.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## eta0: Step-size hyperparameter.
  ## scheduling: How to change the step-size.
  ##  - constant: eta = eta0,
  ##  - optimal: eta = eta0 / pow(1+eta0*regul*it, power),
  ##  - invscaling: eta = eta0 / pow(it, power),
  ##  - pegasos: eta = 1.0 / (regul * it),
  ##  where regul is the Regularization-strength hyperparameter.
  ## power: Hyperparameter for step size scheduling.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  ## shuffle: How to choose one instance: cyclic (false) or random permutation
  ##          (true).
  ## nCalls: Frequency with which callback must be called in the inner loop.
  ##         If nCalls <= 0 or multi-thread case, callback is called per one epoch.
  ## 
  result = SGD[L](
    maxIter: maxIter, eta0: eta0, alpha0: alpha0, alpha: alpha, beta: beta,
    loss: loss, scheduling: scheduling, power: power, it: 1, verbose: verbose,
    tol: tol, shuffle: shuffle, nCalls: nCalls)


proc init*[L](self: SGD[L]) =
  self.it = 1
  if self.verbose > 0: echoHeader(self.maxIter)


proc getEta*[T](self: T, reg: float64): float64 {.inline.} =
  case self.scheduling
  of constant:
    result = self.eta0
  of optimal:
    result = self.eta0 / pow(1.0+self.eta0*reg*float(self.it), self.power)
  of invscaling:
    result = self.eta0 / pow(toFloat(self.it), self.power)
  of pegasos:
    result = 1.0 / (reg * toFloat(self.it))


proc stoppingCriterion*(P: Tensor, w: Vector, intercept: float64,
                        alpha0, alpha, beta, lossVal, viol, tol: float64,
                        verbose, epoch, maxIter: int,
                        isConverged: var bool): bool =
  result = true
  if lossVal.classify == fcNan:
    echo("Loss is NaN. Use smaller learning rate.")
    result = false

  if verbose > 0:
    let reg = regularization(P, w, intercept, alpha0, alpha, beta)
    echoInfo(epoch+1, maxIter, viol, lossVal, reg)

  if viol < tol:
    if verbose > 0:
      echo(fmt"Converged at epoch {epoch}.")
    isConverged = true
    result = false


proc transpose*(P1: var Tensor, P2: Tensor) =
  for order in 0..<P2.shape[0]:
    for j in 0..<P2.shape[1]:
      for s in 0..<P2.shape[2]:
        P1[order, s, j] = P2[order, j, s]


proc finalize*(P: var Tensor, w: var Vector,
               scaling_P, scaling_w: var float64,
               scalings_P, scalings_w: var Vector, fitLinear: bool) =
  if fitLinear:
    w *= scaling_w
    w /= scalings_w
    scaling_w = 1.0
    scalings_w <- 1.0

  for order in 0..<P.shape[0]:
    for j in 0..<P.shape[1]: # dummy features can be skipped
      for s in 0..<P.shape[2]:
        P[order, j, s] *= scaling_P / scalings_P[j]
  scalings_P <- 1.0
  scaling_P = 1.0


proc resetScaling(P: var Tensor, w: var Vector,
                  scaling_P, scaling_w: var float64,
                  scalings_P, scalings_w: var Vector, fitLinear: bool) =
  if fitLinear and scaling_w < 1e-9:
    w *= scaling_w
    w /= scalings_w
    scalings_w <- 1.0
    scaling_w = 1.0

  if scaling_P < 1e-9:
    for order in 0..<P.shape[0]:
      for j in 0..<len(w): # dummy features can be skipped
        for s in 0..<P.shape[2]:
          P[order, j, s] *= scaling_P / scalings_P[j]
    scalings_P <- 1.0
    scaling_P = 1.0


proc lazilyUpdate*(X: RowDataset, i: int, P: var Tensor, w: var Vector,
                   scaling_P, scaling_w: float64,
                   scalings_P, scalings_w: Vector, fitLinear: bool) =
  for order in 0..<P.shape[0]:
    for j in X.getRowIndices(i):
      for s in 0..<P.shape[2]:
        P[order, j, s] *= scaling_P / scalings_P[j]
  if fitLinear:
    for j in X.getRowIndices(i):
      w[j] *= scaling_w / scalings_w[j]


proc computeAnova*(P: Matrix, X: RowDataset, i, degree: int,
                   A: var Matrix): float64 =
  result = 0.0
  let nComponents = P.shape[1]
  # compute anova kernel
  if degree != 2:
    for s in 0..<nComponents:
      A[s, 0] = 1.0
      for t in 1..<degree+1:
        A[s, t] = 0
    for (j, val) in X.getRow(i):
      for s in 0..<nComponents:
        for t in 0..<degree:
          A[s, degree-t] += A[s, degree-t-1] * P[j, s] * val
  else:
    for s in 0..<nComponents:
      A[s, 0] = 1
      A[s, 1] = 0
      A[s, 2] = 0
    for (j, val) in X.getRow(i):
      for s in 0..<nComponents:
        A[s, 1] += val * P[j, s]
        A[s, 2] += (val*P[j, s])^2
    for s in 0..<nComponents:
      A[s, 2] = (A[s, 1]^2 - A[s, 2])/2

  for s in 0..<nComponents:
    result += A[s, degree]


proc computeAnovaDerivative*(P: Matrix, X: RowDataset, i, degree: int,
                             A: Matrix, dA: var Matrix)  =
  # compute derivatives
  if degree != 2:
    for (j, val) in X.getRow(i):
      for s in 0..<P.shape[1]:
        dA[j, s] = val
        for t in 1..<degree:
          dA[j, s] = val * (A[s, t] - P[j, s] * dA[j, s])
  else:
    for (j, val) in X.getRow(i):
      for s in 0..<P.shape[1]:
        dA[j, s] = val * (A[s, 1] - P[j, s]*val)


proc predictWithGrad*(X: RowDataset, i: int, P: Tensor, w: Vector,
                      intercept: float64, A: var Matrix, dA: var Tensor,
                      degree, nAugments: int): float64 =
  result = intercept
  for (j, val) in X.getRow(i):
    result += w[j] * val

  X.addDummyFeature(1.0, nAugments)
  for order in 0..<P.shape[0]:
    result += computeAnova(P[order], X, i, degree-order, A)
    computeAnovaDerivative(P[order], X, i, degree-order, A, dA[order])
  X.removeDummyFeature(nAugments)


proc update*[L](self: SGD[L], X: RowDataset, P: var Tensor, w: var Vector,
                intercept: var float64, df: Tensor, i: int,
                yi, yPred: float64, scaling_P, scaling_w: var float64,
                scalings_P, scalings_w: var Vector, nAugments: int,
                fitLinear, fitIntercept: bool): float64 =
  # update parameters and caches for scaling
  let dL = self.loss.dloss(yi, yPred)
  let eta_w = self.getEta(self.alpha)
  let eta_P = self.getEta(self.beta)

  # Update Parameters
  X.addDummyFeature(1.0, nAugments)
  for order in 0..<P.shape[0]:
    for j in X.getRowIndices(i):
      for s in 0..<P.shape[2]:
        let update = eta_P * (dL*df[order, j, s] + self.beta*P[order, j, s])
        result += abs(update)
        P[order, j, s] -= update
  X.removeDummyFeature(nAugments)

  # for w and intercept
  if fitIntercept:
    let update = self.getEta(self.alpha0) * (dL + self.alpha0 * intercept)
    result += abs(update)
    intercept -= update
  if fitLinear:
    result += fitLinearSGD(w, X, i, self.alpha, dL, eta_w)

  # update caches for scaling
  scaling_P *= (1-eta_P*self.beta)
  scaling_w *= (1-eta_w*self.alpha)
  for j in X.getRowIndices(i):
    scalings_P[j] = scaling_P
    scalings_w[j] = scaling_w
  if nAugments > 0:
    scalings_P[^nAugments..^1] = scaling_P

  # reset scalings in order to avoid numerical error
  resetScaling(P, w, scaling_P, scaling_w, scalings_P, scalings_w, fitLinear)


proc step*[L](self: SGD[L], X: RowDataset, P: var Tensor, w: var Vector,
              intercept: var float64, i: int, yi: float64,
              A: var Matrix, dA: var Tensor, scaling_P, scaling_w: var float64,
              scalings_P, scalings_w: var Vector, degree, nAugments: int,
              fitLinear, fitIntercept: bool, runningLoss, viol: var float64) =
  # synchronize (lazily update) and compute prediction/gradient
  lazilyUpdate(X, i, P, w, scaling_P, scaling_w, scalings_P, scalings_w,
               fitLinear)
  let yPred = predictWithGrad(X, i, P, w, intercept, A, dA, degree, nAugments)
  runningLoss += self.loss.loss(yi, yPred)
  viol += update(self, X, P, w, intercept, dA, i, yi, yPred, scaling_P,
                 scaling_w, scalings_P, scalings_w, nAugments, fitLinear,
                 fitIntercept)
  

proc fit*[L](self: SGD[L], X: RowDataset, y: seq[float64],
             fm: FactorizationMachine,
             callback: (SGD[L], FactorizationMachine)->void = nil) =
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
    nCalls = self.nCalls
  var
    scaling_w = 1.0
    scaling_P = 1.0
    scalings_w = ones([len(fm.w)])
    scalings_P = ones([fm.P.shape[2]])
    A: Matrix = zeros([nComponents, degree+1])
    indices = toSeq(0..<nSamples)
    P: Tensor = zeros([nOrders, fm.P.shape[2], nComponents])
    dA: Tensor = zeros(P.shape)
    isConverged = false

  if not fm.warmstart:
    self.init()

  # copy for fast training
  transpose(P, fm.P)

  for epoch in 0..<self.maxIter:
    var viol = 0.0
    var runningLoss = 0.0
    if X.nCached == X.nSamples and self.shuffle: shuffle(indices)
    for i in indices:
      step(self, X, P, fm.w, fm.intercept, i, y[i], A, dA, scaling_P,
           scaling_w, scalings_P, scalings_w, degree, nAugments, fit_linear,
           fitIntercept, runningLoss, viol)

      if not callback.isNil and nCalls > 0 and self.it mod nCalls == 0:
        finalize(P, fm.w, scaling_P, scaling_w, scalings_P, scalings_w,
                 fitLinear)
        transpose(fm.P, P)
        callback(self, fm)
      inc(self.it)

    # one epoch done
    runningLoss /= float(nSamples)
    if not callback.isNil and nCalls <= 0:
      finalize(P, fm.w, scaling_P, scaling_w, scalings_P, scalings_w,
               fitLinear)
      transpose(fm.P, P)
      callback(self, fm)

    let isContinue = stoppingCriterion(
      P, fm.w, fm.intercept, self.alpha0, self.alpha, self.beta, runningLoss,
      viol, self.tol, self.verbose, epoch, self.maxIter, isConverged)
    if not isContinue: break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(P, fm.w, scaling_P, scaling_w, scalings_P, scalings_w, fitLinear)
  transpose(fm.P, P)