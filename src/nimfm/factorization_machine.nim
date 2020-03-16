import loss, kernels, tensor, fm_base
import sugar, random, sequtils, strutils, parseutils


type
  FitLowerKind* = enum
    explicit = "explicit",
    augment = "augment",
    none = "none"

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
                            ##        nComponents,
                            ##        nFeatures+nAugments)
    lams*: Vector           ## Weights for each base. shape (nComponents)
    w*: Vector              ## Weigths for linear term, shape (nFeatures)
    intercept*: float64     # Intercept term

  FactorizationMachine* = ref FactorizationMachineObj


proc newFactorizationMachine*(task: TaskKind, degree = 2, nComponents = 30,
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
  ## beta: Regularization strength for higher-order weights.
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
  if nComponents < 1:
    raise newException(ValueError, "nComponents < 1.")
  result.nComponents = nComponents
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
  result.lams = ones([result.nComponents])


proc nAugments*[FM](self: FM): int =
  case self.fitLower
  of explicit, none:
    result = 0
  of augment:
    result = if self.fitLinear: self.degree-2 else: self.degree-1


proc nOrders*[FM](self: FM): int =
  if self.degree == 1:
    result = 0
  else:
    case self.fitLower
    of explicit:
      result = self.degree-1
    of none, augment:
      result = 1


proc decisionFunction*[FM, Dataset](self: FM, X: Dataset): seq[float64] =
  ## Returns the model outputs as seq[float64].
  self.checkInitialized()
  let nSamples: int = X.nSamples
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
    for s in 0..<self.P.shape[1]:
      anova(X, self.P[order], A, self.degree-order, s, nAugments)
      for i in 0..<nSamples:
        result[i] += self.lams[s]*A[i, self.degree-order]


proc init*[Dataset, FM](self: FM, X: Dataset, force=false) =
  ## Initializes the factorization machine.
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


proc dump*(self: FactorizationMachine, fname: string) =
  ## Dumps the fitted factorization machine.
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


proc load*(fm: var FactorizationMachine, fname: string, warmStart: bool) =
  ## Loads the fitted factorization machine.
  new(fm)
  var f: File = open(fname, fmRead)
  var nFeatures: int
  fm.task = parseEnum[TaskKind](f.readLine().split(" ")[1])
  discard parseInt(f.readLine().split(" ")[1], nFeatures, 0)
  discard parseInt(f.readLine().split(" ")[1], fm.degree, 0)
  discard parseInt(f.readLine().split(" ")[1], fm.nComponents, 0)
  discard parseFloat(f.readLine().split(" ")[1], fm.alpha0, 0)
  discard parseFloat(f.readLine().split(" ")[1], fm.alpha, 0)
  discard parseFloat(f.readLine().split(" ")[1], fm.beta, 0)
  fm.loss = parseEnum[LossFunctionKind](f.readLine().split(" ")[1])
  fm.fitLower = parseEnum[FitLowerKind](f.readLine().split(" ")[1])
  fm.fitIntercept = parseBool(f.readLine().split(" ")[1])
  fm.fitLinear = parseBool(f.readLine().split(" ")[1])
  fm.warmStart = warmStart
  discard parseInt(f.readLine().split(" ")[1], fm.randomState, 0)
  discard parseFloat(f.readLine().split(" ")[1], fm.scale, 0)

  let nOrders = fm.nOrders
  let nAugments = fm.nAugments
  var i = 0
  var val: float64
  fm.P = zeros([nOrders, fm.nComponents, nFeatures+nAugments])
  for order in 0..<nOrders:
    discard f.readLine() # read "P[order]:" and discard it
    for s in 0..<fm.nComponents:
      let line = f.readLine()
      var i = 0
      var val: float64
      for j in 0..<nFeatures:
        i.inc(parseFloat(line, val, i))
        i.inc()
        fm.P[order, s, j] = val
  fm.w = zeros([nFeatures])
  discard f.readLine()
  let line = f.readLine()
  i = 0
  for j in 0..<nFeatures:
    i.inc(parseFloat(line, val, i))
    i.inc()
    fm.w[j] = val
  discard parseFloat(f.readLine().split(" ")[1], fm.intercept, 0)
  f.close()
  fm.isInitalized = true
