import ../kernels, ../tensor/tensor, fm_base, ../dataset
import random, sequtils, strutils, parseutils, typetraits


type
  FitLowerKind* = enum
    explicit = "explicit",
    augment = "augment",
    none = "none"

  FactorizationMachineObj* = object
    task*: TaskKind         ## regression or classification.
    degree*: int            ## Degree of the polynomial.
    nComponents*: int       ## Number of basis vectors (rank hyperparameter).
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
    isInitialized*: bool
    P*: Tensor              ## Weights for the polynomial.
                            ## shape (degree-2 or degree-1 or 1,
                            ##        nComponents,
                            ##        nFeatures+nAugments)
    lams*: Vector           ## Weights for vectors in basis.
                            ## shape: (nComponents)
    w*: Vector              ## Weigths for linear term, shape: (nFeatures)
    intercept*: float64     ## Intercept term.

  FactorizationMachine* = ref FactorizationMachineObj


proc newFactorizationMachine*(
  task: TaskKind, degree = 2, nComponents = 30, fitLower = explicit, 
  fitIntercept = true, fitLinear = true, warmStart = false, randomState = 1,
  scale = 0.01): FactorizationMachine =
  ## Create a new FactorizationMachine.
  ## task: classification or regression.
  ## degree: Degree of the polynomial (= the order of feature interactions).
  ## nComponents: Number of basis vectors (rank hyperparameter).
  ## fitLower: Whether and how to fit lower-order terms.
  ##   - explicit: fits a seperate weights for each lower order.
  ##   - augment: adds the dummy feature for each feature vector and fits\
  ##              only weights for degree-order. It uses lower order\
  ##              interactions implicitly.
  ##   - none: learns only the weights for degree-order.
  ## fitIntercept: Whether to fit intercept (a.k.a bias) term or not.
  ## fitLinear: Whether to fit linear term or not.
  ## warmStart: Whether to do warwm start fitting or not.
  ## randomState: The seed of the pseudo random number generator.
  ## scale: The scale (a.k.a std) of the Gaussian distribution for initialization
  ##        of higher-order weights.
  new(result)
  result.task = task
  if degree < 1:
    raise newException(ValueError, "degree < 1.")
  result.degree = degree
  if nComponents < 1:
    raise newException(ValueError, "nComponents < 1.")
  result.nComponents = nComponents
  result.fitLower = fitLower
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.warmStart = warmStart
  result.randomState = randomState
  result.scale = scale
  result.isInitialized = false
  result.lams = ones([result.nComponents])


proc nAugments*(self: FactorizationMachine): int =
  case self.fitLower
  of explicit, none:
    result = 0
  of augment:
    result = if self.fitLinear: self.degree-2 else: self.degree-1


proc nOrders*(self: FactorizationMachine): int =
  if self.degree == 1:
    result = 0
  else:
    case self.fitLower
    of explicit:
      result = self.degree-1
    of none, augment:
      result = 1


proc decisionFunction*[Dataset](self: FactorizationMachine, X: Dataset): seq[float64] =
  ## Returns the model outputs as seq[float64].
  self.checkInitialized()

  let nSamples: int = X.nSamples
  var A = zeros([nSamples, self.degree+1])
  result = newSeqWith(nSamples, 0.0)

  linear(X, self.w, result)
  for i in 0..<nSamples:
    result[i] += self.intercept

  X.addDummyFeature(1.0, self.nAugments)
  let nFeatures = X.nFeatures
  if (nFeatures != self.P.shape[2]):
    raise newException(ValueError, "Invalid nFeatures.")
  for order in 0..<self.nOrders:
    for s in 0..<self.P.shape[1]:
      anova(X, self.P[order], A, self.degree-order, s)
      for i in 0..<nSamples:
        result[i] += self.lams[s]*A[i, self.degree-order]
  
  X.removeDummyFeature(self.nAugments)


proc init*[Dataset](self: FactorizationMachine, X: Dataset, force=false) =
  ## Initializes the factorization machine.
  ## self will not be initialized if force=false, fm is already initialized,
  ## and warmStart=true.
  if force or not (self.warmStart and self.isInitialized):
    let nFeatures: int = X.nFeatures
    randomize(self.randomState)

    self.w = zeros([nFeatures])
    let nOrders = self.nOrders
    let nAugments = self.nAugments
    self.P = randomNormal([nOrders, self.nComponents, nFeatures+nAugments],
                          scale = self.scale)
    self.intercept = 0.0
  self.isInitialized = true


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
  f.writeLine("fitLower: ", self.fitLower)
  f.writeLine("fitIntercept: ", self.fitIntercept)
  f.writeLine("fitLinear: ", self.fitLinear)
  f.writeLine("randomState: ", self.randomState)
  f.writeLine("scale: ", self.scale)
  f.writeLine("lams:")
  f.writeLine(self.lams.join(" "))
  for order in 0..<self.P.shape[0]:
    f.writeLine("P[", order, "]:")
    for s in 0..<nComponents:
      f.writeLine(self.P[order, s].join(" "))
  f.writeLine("w:")
  f.writeLine(self.w.join(" "))
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
  # read lams
  discard f.readLine() # read "lams:"
  let line_lams = f.readLine()
  i = 0
  fm.lams = zeros([fm.nComponents])
  for s in 0..<fm.nComponents:
    i.inc(parseFloat(line_lams, val, i))
    i.inc()
    fm.lams[s] = val
  
  # read P
  fm.P = zeros([nOrders, fm.nComponents, nFeatures+nAugments])
  for order in 0..<nOrders:
    discard f.readLine() # read "P[order]:" and discard it
    for s in 0..<fm.nComponents:
      let line = f.readLine()
      i = 0
      for j in 0..<nFeatures+nAugments:
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
  fm.isInitialized = true
