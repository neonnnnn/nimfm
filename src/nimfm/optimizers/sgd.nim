import ../loss, ../dataset, ../tensor, ../factorization_machine
import fit_linear, base
import sequtils, math, random, strformat, strutils

type
  SchedulingKind* = enum
    constant = "constant",
    optimal = "optimal",
    invscaling = "invscaling",
    pegasos = "pegasos"

  SGD* = ref object of BaseCSROptimizer
    eta0: float64
    scheduling: SchedulingKind
    power: float64
    it: int
    shuffle: bool


proc newSGD*(eta0 = 0.01, scheduling = optimal, power = 1.0, maxIter = 100,
             verbose = true, tol = 1e-3, shuffle = true): SGD =
  ## Creates new SGD.
  ## eta0: Step-size parameter.
  ## scheduling: How to change the step-size.
  ##  - constant: eta = eta0,
  ##  - optimal: eta = eta0 / pow(1+eta0*regul*it, power),
  ##  - invscaling: eta = eta0 / pow(it, power),
  ##  - pegasos: eta = 1.0 / (regul * it),
  ##  where regul is the regularization strength hyper-parameter.
  ## power: Hyper-parameter for step size scheduling.
  ## maxIter: Maximum number of iteration. In one iteration. \
  ## all parameters are updated once by using all samples.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyper-parameter for stopping criterion.
  ## shuffle: How to choose one instance: cyclic (false) or random permutation
  ##          (true).
  result = SGD(eta0: eta0, scheduling: scheduling, power: power, it: 1,
               maxIter: maxIter, tol: tol, verbose: verbose,
               shuffle: shuffle)


proc getEta(self: SGD, reg: float64): float64 {.inline.} =
  case self.scheduling
  of constant:
    result = self.eta0
  of optimal:
    result = self.eta0 / pow(1.0+self.eta0*reg*float(self.it), self.power)
  of invscaling:
    result = self.eta0 / pow(toFloat(self.it), self.power)
  of pegasos:
    result = 1.0 / (reg * toFloat(self.it))


proc computeAnova(P: Tensor, X: CSRDataset,
                  i, degree, order, nAugments: int, A: var Matrix,
                  dA: var Tensor, dAcache: var Vector,
                  PScaling: float64, PScalings: seq[float64]): float64 =
  result = 0.0
  let
    nComponents = P.shape[2]
    nFeatures = X.nFeatures
  for s in 0..<nComponents:
    A[s, 0] = 1.0
    for t in 1..<degree+1:
      A[s, t] = 0

  # compute anova kernel
  for (j, val) in X.getRow(i):
    for s in 0..<nComponents:
      P[order, j, s] *= PScaling / PScalings[j]
      for t in 0..<degree:
        A[s, degree-t] += A[s, degree-t-1] * P[order, j, s] * val
  # for augmented features
  for j in nFeatures..<(nFeatures+nAugments):
    for s in 0..<nComponents:
      for t in 0..<degree:
        A[s, degree-t] += A[s, degree-t-1] * P[order, j, s]

  for s in 0..<nComponents:
    result += A[s, degree]

  # compute derivatives
  for (j, val) in X.getRow(i):
    for s in 0..<nComponents:
      dAcache[0] = val
      for t in 1..<degree:
        dAcache[t] = val * (A[s, t] - P[order, j, s] * dAcache[t-1])
      dA[order, j, s] = dAcache[degree-1]
  
  # for augmented features
  for j in nFeatures..<(nFeatures+nAugments):
    for s in 0..<nComponents:
      dAcache[0] = 1.0
      for t in 1..<degree:
        dAcache[t] = A[s, t] - P[order, j, s] * dAcache[t-1]
      dA[order, j, s] = dAcache[degree-1]


proc computeAnovaDeg2(P: Tensor, X: CSRDataset,
                      i, order, nAugments: int, A: var Matrix,
                      dA: var Tensor, PSCaling: float64,
                      PSCalings: seq[float64]): float64 =
  result = 0.0
  let
    nComponents = P.shape[2]
    nFeatures = X.nFeatures
  for s in 0..<nComponents:
    A[s, 0] = 1
    A[s, 1] = 0
    A[s, 2] = 0
  # compute anova kernel
  for (j, val) in X.getRow(i):
    for s in 0..<nComponents:
      P[order, j, s] *= PSCaling / PSCalings[j]
      A[s, 1] += val * P[order, j, s]
      A[s, 2] += (val*P[order, j, s])^2
  # for augmented features
  if nAugments == 1:
    for s in 0..<nComponents:
      A[s, 1] += P[order, nFeatures, s]
      A[s, 2] += P[order, nFeatures, s]^2

  for s in 0..<nComponents:
    A[s, 2] = (A[s, 1]^2 - A[s, 2])/2
    result += A[s, 2]

  # compute derivatives
  for (j, val) in X.getRow(i):
    for s in 0..<nComponents:
      dA[order, j, s] = val * (A[s, 1] - P[order, j, s]*val)
  # for augmented features
  if nAugments == 1:
    for s in 0..<nComponents:
      dA[order, nFeatures, s] = A[s, 1] - P[order, nFeatures, s]


proc fit*(self: SGD, X: CSRDataset, y: seq[float64],
          fm: var FactorizationMachine) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  fm.init(X)
  let y = fm.checkTarget(y)
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    alpha0 = fm.alpha0
    alpha = fm.alpha
    beta = fm.beta
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nAugments = fm.nAugments
    loss = newLossFunction(fm.loss)
  var
    wScaling = 1.0
    PScaling = 1.0
    wScalings = newSeqWith(nFeatures, 1.0)
    PScalings = newSeqWith(nFeatures, 1.0)
    A: Matrix = zeros([nComponents, degree+1])
    dAcache: Vector = zeros([degree])
    indices = toSeq(0..<nSamples)
    P: Tensor = zeros([nOrders, nFeatures+nAugments, nComponents])
    dA: Tensor = zeros(P.shape)
    isConverged = false
  
  if not fm.warmstart:
    self.it = 1
  # copy for fast training
  for order in 0..<nOrders:
    for s in 0..<nComponents:
      for j in 0..<nFeatures+nAugments:
        P[order, j, s] = fm.P[order, s, j]
  for epoch in 0..<self.maxIter:
    var viol = 0.0
    var runningLoss = 0.0
    if self.shuffle: shuffle(indices)

    for i in indices:
      let wEta = self.getEta(alpha)
      let PEta = self.getEta(beta)
      # synchronize (lazily update) and compute prediction
      var yPred = fm.intercept
      for (j, val) in X.getRow(i):
        fm.w[j] *= wScaling / wScalings[j]
        yPred += fm.w[j] * val
      for order in 0..<nOrders:
        if (degree-order) > 2:
          yPred += computeAnova(P, X, i, degree-order, order, nAugments, 
                                A, dA, dACache, PScaling, PScalings)
        else:
          yPred += computeAnovaDeg2(P, X, i, order, nAugments, A,
                                    dA, PScaling, PScalings)
      runningLoss += loss.loss(yPred, y[i])

      # update parameters
      let dL = loss.dloss(yPred, y[i])

      if fitIntercept:
        let update = self.getEta(alpha0) * (dL + alpha0 * fm.intercept)
        viol += abs(update)
        fm.intercept -= update

      if fitLinear:
        viol += fitLinearSGD(fm.w, X, alpha, dL, wEta, i)
      
      for order in 0..<nOrders:
        for (j, _) in X.getRow(i):
          for s in 0..<nComponents:
            let update = PEta * (dL*dA[order, j, s] + beta*P[order, j, s])
            viol += abs(update)
            P[order, j, s] -= update
        # for augmented features
        for j in nFeatures..<(nFeatures+nAugments):
          for s in 0..<nComponents:
            let update = PEta * (dL*dA[order, j, s] + beta*P[order, j, s])
            viol += abs(update)
            P[order, j, s] -= update
      
      # update caches for scaling
      wScaling *= (1-wEta*alpha)
      PScaling *= (1-PEta*beta)
      for (j, _) in X.getRow(i):
        wScalings[j] = wScaling
        PScalings[j] = PScaling
      
      # reset scalings in order to avoid numerical error
      if fitLinear and wScaling < 1e-9:
        for j in 0..<nFeatures:
          fm.w[j] *= wScaling / wScalings[j]
          wScalings[j] = 1.0
        wScaling = 1.0
      if PScaling < 1e-9:
        for order in 0..<nOrders:
          for j in 0..<nFeatures:
            for s in 0..<nComponents:
              P[order, j, s] *= PScaling / PScalings[j]
        for j in 0..<nFeatures:
          PScalings[j] = 1.0
        PScaling = 1.0
      
      inc(self.it)

    # one epoch done
    if self.verbose:
      runningLoss /= float(nSamples)
      stdout.write(fmt"Epoch: {align($(epoch+1), len($self.maxIter))}")
      stdout.write(fmt"   Violation: {viol:1.4e}")
      stdout.write(fmt"   Loss: {runningLoss:1.4e}")
      stdout.write("\n")
      stdout.flushFile()
    if viol < self.tol:
      if self.verbose: echo("Converged at epoch ", epoch, ".")
      isConverged = true
      break
  
  if not isConverged and self.verbose:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  if fitLinear:
    for j in 0..<nFeatures:
      fm.w[j] *= wScaling / wScalings[j]
  for order in 0..<nOrders:
    for j in 0..<nFeatures:
      for s in 0..<nComponents:
        P[order, j, s] *= PScaling / PScalings[j]
        fm.P[order, s, j] = P[order, j, s]
    for j in nFeatures..<nFeatures+nAugments:
      for s in 0..<nComponents:
        fm.P[order, s, j] = P[order, j, s]
