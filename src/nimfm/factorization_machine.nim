import loss, kernels, tensor, metrics
import math, sugar, random, sequtils, strutils, parseutils

type
  FitLowerKind* = enum
    explicit = "explicit",
    augment = "augment",
    none = "none"

  TaskKind* = enum
    regression = "r",
    classification = "c"

  FactorizationMachineObj* = object
    task*: TaskKind         ## regression or classification.
    degree*: int            ## Degree of the polynomial.
    nComponents*: int       ## Number of basis vectors (rank hyper-parameter).
    alpha0*: float64        ## Regularization strength for intercept.
    alpha*: float64         ## Regularization strength for linear term.
    beta*: float64          ## Regularization strengt for higher-order weights.
    loss*: LossFunctionKind ## Loss function.
                            ## Squared, SquaredHinge, or Logistic.
    fitLower*: FitLowerKind ## Whether and how to fit lower-order terms.
                            ##   - explicit: fits a seperate weights for each
                            ##               lower order.
                            ##   - augment: adds the dummy feature fits only
                            ##              weights for degree-order. It uses
                            ##              lower order interactions
                            ##              implicitly.
                            ##   - none: learns only the weights for
                            ##           degree-order.
    fitIntercept*: bool     ## Whether to fit intercept (a.k.a bias) term.
    fitLinear*: bool        ## Whether to fit linear term.
    warmStart*: bool        ## Whether to do warwm start fitting.
    randomState*: int       ## The seed of the pseudo random number generator.
    scale*: float64         ## The scale (a.k.a std) of Normal distribution for
                            ## initialization of higher-order weights.
    isInitalized*: bool
    P*: Tensor              ## Weights for the polynomial.
                            ## shape (degree-2 or degree-1 or 1,
                            ##        n_components,
                            ##        n_features+nAugments)
    w*: Vector              ## Weigths for linear term, shape (n_features)
    intercept*: float64     # Intercept term

  FactorizationMachine* = ref FactorizationMachineObj

  NotFittedError = object of Exception


proc checkInitialized*(self: FactorizationMachine) =
  if not self.isInitalized:
    raise newException(NotFittedError, "Factorization machines is not fitted.")


proc newFactorizationMachine*(task: TaskKind, degree = 2, n_components = 30,
                              alpha0 = 1e-6, alpha = 1e-3, beta = 1e-3,
                              loss = SquaredHinge, fitLower = explicit,
                              fitIntercept = true,
                              fitLinear = true, warmStart = false,
                              randomState = 1, scale = 0.1):
                              FactorizationMachine =
  ## Create a new FactorizationMachine.
  ## task: classification or regression.
  ## degree: Degree of the polynomial (= the order of feature interactions).
  ## nComponents: Number of basis vectors (a.k.a rank hyper-parameter).
  ## alpha0: Regularization strength for intercept.
  ## alpha: Regularization strength for linear term.
  ## beta: Regularization strengt for higher-order weights.
  ## loss: Loss function.
  ##   - Squared: 1/2 (y-p)**2,
  ##   - SquaredHinge: (max(0, 1-y*p))**2,
  ##   - Logistic: log (1+exp(-y*p)), where p is the predicted value.
  ## fitLower: Whether and how to fit lower-order terms.
  ##   - explicit: fits a seperate weights for each lower order.
  ##   - augment: adds the dummy feature for each feature vectors and fits\
  ##              only weights for degree-order. It uses lower order\
  ##              interactions implicitly.
  ##   - none: learns only the weights for degree-order.
  ## fitIntercept: Whether to fit intercept (a.k.a bias) term or not.
  ## fitLinear: Whether to fit linear term or not.
  ## warmStart: Whether to do warwm start fitting or not.
  ## randomState: The seed of the pseudo random number generator.
  ## scale: The scale (a.k.a std) of the Normal distribution for initialization
  ##        of higher-order weights.
  new(result)
  result.task = task
  if degree < 1:
    raise newException(ValueError, "degree < 1.")
  result.degree = degree
  if n_components < 1:
    raise newException(ValueError, "nComponents < 1.")
  result.n_components = n_components
  if alpha0 < 0 or alpha < 0 or beta < 0:
    raise newException(ValueError, "Regularization strength < 0.")
  result.alpha0 = alpha0
  result.alpha = alpha
  result.beta = beta
  case task
  of regression: result.loss = Squared
  of classification: result.loss = loss
  result.fitLower = fitLower
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.warmStart = warmStart
  result.randomState = randomState
  result.scale = scale
  result.isInitalized = false


proc nAugments*(self: FactorizationMachine): int =
  case self.fitLower
  of explicit, none:
    result = 0
  of augment:
    result = if self.fitLinear: self.degree-2 else: self.degree-1


proc nOrders(self: FactorizationMachine): int =
  if self.degree == 1:
    result = 0
  else:
    case self.fitLower
    of explicit:
      result = self.degree-1
    of none, augment:
      result = 1


proc decisionFunction*[Dataset](self: FactorizationMachine, X: Dataset):
                                seq[float64] =
  ## Returns the model outputs as seq[float64].
  self.checkInitialized()
  let nSamples: int = X.n_samples
  let nComponents: int = self.nComponents
  let nFeatures = X.nFeatures
  var A = zeros([nSamples, self.degree+1])
  result = newSeqWith(nSamples, 0.0)

  linear(X, self.w, result)
  for i in 0..<nSamples:
    result[i] += self.intercept

  var nAugments = self.nAugments
  if (nAugments + nFeatures != self.P.shape[2]):
    raise newException(ValueError, "Invalid nFeatures.")
  for order in 0..<self.nOrders:
    for s in 0..<nComponents:
      anova(X, self.P, A, self.degree-order, order, s, nAugments)
      for i in 0..<nSamples:
        result[i] += A[i, self.degree-order]


proc predict*[Dataset](self: FactorizationMachine, X: Dataset): seq[int] =
  ## Returns the sign vector of the model outputs as seq[int].
  result = decisionFunction(self, X).map(x=>sgn(x))


proc init*[Dataset](self: FactorizationMachine, X: Dataset, force=false) =
  ## Initialize the factorization machines.
  ## If force=false, fm is already initialized, and warmStart=true,
  ## fm will not be initialized.
  if force or not (self.warmStart and self.isInitalized):
    let nFeatures: int = X.nFeatures
    randomize(self.randomState)

    self.w = zeros([nFeatures])
    let nOrders = self.nOrders
    let nAugments = self.nAugments
    self.P = randomNormal([nOrders, self.nComponents, nFeatures+nAugments],
                          scale = self.scale)
    self.intercept = 0.0
  self.isInitalized = true


proc checkTarget*(self: FactorizationMachine, y: seq[SomeNumber]):
                  seq[float64] =
  case self.task
  of classification:
    result = y.map(x => float(sgn(x)))
  of regression:
    result = y.map(x => float(x))


proc score*[Dataset](self: FactorizationMachine, X: Dataset,
                     y: seq[float64]): float64 =
  ## Returns the score between the model outputs and true targets.
  ## Computes root mean squared error when task=regression (lower is better).
  ## Computes accuracy when task=classification (higher is better). 
  let yPred = self.decisionFunction(X)
  case self.task
  of regression:
    result = rmse(yPred, y)
  of classification:
    result = accuracy(yPred.map(x=>sgn(x)), y.map(x=>sgn(x)))



proc dump*(self: FactorizationMachine, fname: string) =
  ## Dumps the fitted factorization machines.
  self.checkInitialized()
  let nComponents = self.P.shape[1]
  let nFeatures = self.P.shape[2] - self.nAugments
  var f: File = open(fname, fmWrite)
  f.writeLine("task: ", $self.task)
  f.writeLine("nFeatures: ", nFeatures)
  f.writeLine("degree: ", self.degree)
  f.writeLine("nComponents: ", self.nComponents)
  f.writeLine("alpha0: ", self.alpha0)
  f.writeLine("alpha: ", self.alpha)
  f.writeLine("beta: ", self.beta)
  f.writeLine("loss: ", self.loss)
  f.writeLine("fitLower: ", self.fitLower)
  f.writeLine("fitIntercept: ", self.fitIntercept)
  f.writeLine("fitLinear: ", self.fitLinear)
  f.writeLine("randomState: ", self.randomState)
  f.writeLine("scale: ", self.scale)
  var params: seq[float64] = newSeq[float64](nFeatures)
  for order in 0..<self.P.shape[0]:
    f.writeLine("P[", order, "]:")
    for s in 0..<nComponents:
      for j in 0..<nFeatures:
        params[j] = self.P[order, s, j]
      f.writeLine(params.join(" "))
  f.writeLine("w:")
  for j in 0..<nFeatures:
    params[j] = self.w[j]
  f.writeLine(params.join(" "))
  f.writeLine("intercept: ", self.intercept)
  f.close()


proc load*(fname: string, warmStart: bool): FactorizationMachine =
  ## Loads the fitted factorization machines.
  new(result)
  var f: File = open(fname, fmRead)
  var nFeatures: int
  result.task = parseEnum[TaskKind](f.readLine().split(" ")[1])
  discard parseInt(f.readLine().split(" ")[1], nFeatures, 0)
  discard parseInt(f.readLine().split(" ")[1], result.degree, 0)
  discard parseInt(f.readLine().split(" ")[1], result.nComponents, 0)
  discard parseFloat(f.readLine().split(" ")[1], result.alpha0, 0)
  discard parseFloat(f.readLine().split(" ")[1], result.alpha, 0)
  discard parseFloat(f.readLine().split(" ")[1], result.beta, 0)
  result.loss = parseEnum[LossFunctionKind](f.readLine().split(" ")[1])
  result.fitLower = parseEnum[FitLowerKind](f.readLine().split(" ")[1])
  result.fitIntercept = parseBool(f.readLine().split(" ")[1])
  result.fitLinear = parseBool(f.readLine().split(" ")[1])
  result.warmStart = warmStart
  discard parseInt(f.readLine().split(" ")[1], result.randomState, 0)
  discard parseFloat(f.readLine().split(" ")[1], result.scale, 0)

  let nOrders = result.nOrders
  let nAugments = result.nAugments
  var i = 0
  var val: float64
  result.P = zeros([nOrders, result.nComponents, nFeatures+nAugments])
  for order in 0..<nOrders:
    discard f.readLine() # read "P[order]:" and discard it
    for s in 0..<result.nComponents:
      let line = f.readLine()
      var i = 0
      var val: float64
      for j in 0..<nFeatures:
        i.inc(parseFloat(line, val, i))
        i.inc()
        result.P[order, s, j] = val
  result.w = zeros([nFeatures])
  discard f.readLine()
  let line = f.readLine()
  i = 0
  for j in 0..<nFeatures:
    i.inc(parseFloat(line, val, i))
    i.inc()
    result.w[j] = val
  discard parseFloat(f.readLine().split(" ")[1], result.intercept, 0)
  f.close()
  result.isInitalized = true
