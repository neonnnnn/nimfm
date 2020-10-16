import ../dataset, ../tensor/tensor, ../models/factorization_machine, ../loss
from ../models/fm_base import checkTarget
import fit_linear, optimizer_base, utils
import sequtils, math, random, sugar

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
    it: int
    shuffle: bool
    nCalls: int


proc newSGD*[L](maxIter=100, alpha0=1e-6, alpha=1e-3, beta=1e-3,
                loss: L = newSquared(), eta0 = 0.01, scheduling = optimal,
                power = 1.0, verbose = 1, tol = 1e-3, shuffle = true,
                nCalls = -1): SGD[L] =
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
  ##         If nCalls <= 0, callback is called per one epoch.
  ## 
  result = SGD[L](
    maxIter: maxIter, eta0: eta0, alpha0: alpha0, alpha: alpha, beta: beta,
    loss: loss, scheduling: scheduling, power: power, it: 1, verbose: verbose,
    tol: tol, shuffle: shuffle, nCalls: nCalls)


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


proc finalize*(fm: FactorizationMachine, P: Tensor, scaling_P: var float64,
               scalings_P: var Vector, scaling_w: var float64, 
               scalings_w: var Vector, fitLinear: bool) =
  if fitLinear:
    fm.w *= scaling_w
    fm.w /= scalings_w
    scaling_w = 1.0
    scalings_w <- 1.0

  for order in 0..<P.shape[0]:
    for j in 0..<P.shape[1]: # dummy features can be skipped
      for s in 0..<P.shape[2]:
        P[order, j, s] *= scaling_P / scalings_P[j]
        fm.P[order, s, j] = P[order, j, s]
  scalings_P <- 1.0
  scaling_P = 1.0


proc resetScaling(P: var Tensor, w: var Vector, scaling_P: var float64,
                  scalings_P: var Vector, scaling_w: var float64,
                  scalings_w: var Vector, fitLinear: bool) =
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


proc lazilyUpdate(X: RowDataset, i: int, P: var Tensor, w: var Vector,
                  scaling_P: float64, scalings_P: Vector, scaling_w: float64,
                  scalings_w: Vector, fitLinear: bool) =
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
  let
    nComponents = P.shape[1]
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
    alpha0 = self.alpha0
    alpha = self.alpha
    beta = self.beta
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
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
    self.it = 1

  # copy for fast training
  for order in 0..<fm.P.shape[0]:
    P[order] = fm.P[order].T
  if self.verbose > 0: # echo header
    echoHeader(self.maxIter)

  for epoch in 0..<self.maxIter:
    var viol = 0.0
    var runningLoss = 0.0
    if self.shuffle: shuffle(indices)

    for i in indices:
      # synchronize (lazily update) and compute prediction
      lazilyUpdate(X, i, P, fm.w, scaling_P, scalings_P, scaling_w, scalings_w,
                   fitLinear)
      let yPred = predictWithGrad(X, i, P, fm.w, fm.intercept, A, dA, degree,
                                  nAugments)
      runningLoss += self.loss.loss(y[i], yPred)

      # update parameters and caches for scaling
      let dL = self.loss.dloss(y[i], yPred)
      let eta_w = self.getEta(alpha)
      let eta_P = self.getEta(beta)

      # for P
      X.addDummyFeature(1.0, nAugments)
      for order in 0..<nOrders:
        for j in X.getRowIndices(i):
          for s in 0..<nComponents:
            let update = eta_P * (dL*dA[order, j, s] + beta*P[order, j, s])
            viol += abs(update)
            P[order, j, s] -= update
      X.removeDummyFeature(nAugments)

      # for w and intercept
      if fitIntercept:
        let update = self.getEta(alpha0) * (dL + alpha0 * fm.intercept)
        viol += abs(update)
        fm.intercept -= update
      if fitLinear:
        viol += fitLinearSGD(fm.w, X, i, alpha, dL, eta_w)

      # update caches for scaling
      scaling_P *= (1-eta_P*beta)
      scaling_w *= (1-eta_w*alpha)
      for j in X.getRowIndices(i):
        scalings_P[j] = scaling_P
        scalings_w[j] = scaling_w
      if nAugments > 0:
        scalings_P[^nAugments..^1] = scaling_P

      # reset scalings in order to avoid numerical error
      resetScaling(P, fm.w, scaling_P, scalings_P, scaling_w, scalings_w,
                   fitLinear)
      
      if self.nCalls > 0 and self.it mod self.nCalls == 0:
        if not callback.isNil:
          finalize(fm, P, scaling_P, scalings_P, scaling_w, scalings_w,
                   fitLinear)
          callback(self, fm)

      inc(self.it)

    # one epoch done
    if not callback.isNil and self.nCalls <= 0:
      finalize(fm, P, scaling_P, scalings_P, scaling_w, scalings_w, fitLinear)
      callback(self, fm)

    if runningLoss.classify == fcNan:
      echo("Loss is NaN. Use smaller learning rate.")
      break

    if self.verbose > 0:
      runningLoss /= float(nSamples)
      let reg = regularization(P, fm.w, fm.intercept, alpha0, alpha, beta)
      echoInfo(epoch+1, self.maxIter, viol, runningLoss, reg)

    if viol < self.tol:
      if self.verbose > 0:
        echo("Converged at epoch ", epoch, ".")
      isConverged = true
      break
    
  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(fm, P, scaling_P, scalings_P, scaling_w, scalings_w, fitLinear)