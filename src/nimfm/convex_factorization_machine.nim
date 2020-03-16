import loss, tensor, kernels, fm_base
import strutils, parseutils, sequtils, sugar, algorithm


type
  ConvexFactorizationMachineObj* = object
    task*: TaskKind         ## regression or classification.
    degree*: int            ## Degree of the polynomial, 2.
    maxComponents*: int     ## Maximum number of basis vectors.
    alpha0*: float64        ## Regularization strength for intercept.
    alpha*: float64         ## Regularization strength for linear term.
    beta*: float64          ## Regularization strengt for higher-order weights.
    loss*: LossFunctionKind ## Loss function.
                            ## Squared, SquaredHinge, or Logistic.
    fitIntercept*: bool     ## Whether to fit intercept (a.k.a bias) term.
    fitLinear*: bool        ## Whether to fit linear term.
    ignoreDiag*: bool       ## Whether ignored diag (FM) or not (PN).
    warmStart*: bool        ## Whether to do warwm start fitting.
    randomState*: int       ## The seed of the pseudo random number generator.
    isInitalized*: bool
    P*: Matrix              ## Weights for the polynomial.
                            ## shape (nComponents, nFeatures)
    lams*: Vector           ## Weights for each base. shape (nComponents, )
    w*: Vector              ## Weigths for linear term, shape (nFeatures)
    intercept*: float64     # Intercept term

  ConvexFactorizationMachine* = ref ConvexFactorizationMachineObj


proc newConvexFactorizationMachine*(
  task: TaskKind, maxComponents = 30, alpha0 = 1e-6, alpha = 1e-3,
  beta = 1e-5, loss = SquaredHinge, fitIntercept = true, fitLinear = true,
  ignoreDiag=true, warmStart = false): ConvexFactorizationMachine =
  ## Create a new ConvexFactorizationMachine.
  ## task: classification or regression.
  ## maxComponents: Maximum number of basis vectors.
  ## alpha0: Regularization strength for intercept.
  ## alpha: Regularization strength for linear term.
  ## beta: Regularization strength for higher-order weights.
  ## loss: Loss function.
  ##   - Squared: 1/2 (y-p)**2,
  ##   - SquaredHinge: (max(0, 1-y*p))**2,
  ##   - Logistic: log (1+exp(-y*p)), where p is the predicted value.
  ## fitIntercept: Whether to fit intercept (a.k.a bias) term or not.
  ## fitLinear: Whether to fit linear term or not.
  ## warmStart: Whether to do warwm start fitting or not.
  new(result)
  result.task = task
  result.degree = 2
  if maxComponents < 1:
    raise newException(ValueError, "maxComponents < 1.")
  result.maxComponents = maxComponents
  if alpha0 < 0 or alpha < 0 or beta < 0:
    raise newException(ValueError, "Regularization strength < 0.")
  result.alpha0 = alpha0
  result.alpha = alpha
  result.beta = beta
  case task
  of regression: result.loss = Squared
  of classification: result.loss = loss
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.ignoreDiag = ignoreDiag
  result.warmStart = warmStart
  result.degree = 2
  result.isInitalized = false
  result.lams = zeros([0])


proc init*[Dataset](self: ConvexFactorizationMachine, X: Dataset, force=false) =
  ## Initializes the factorization machine.
  ## If force=false, fm is already initialized, and warmStart=true,
  ## fm will not be initialized.
  if force or not (self.warmStart and self.isInitalized):
    let nFeatures: int = X.nFeatures
    self.w = zeros([nFeatures])
    self.P = zeros([0, nFeatures])
    self.intercept = 0.0
  self.isInitalized = true


proc decisionFunction*[Dataset](self: ConvexFactorizationMachine, 
                                X: Dataset): seq[float64] =
  ## Returns the model outputs as seq[float64].
  self.checkInitialized()
  let nSamples: int = X.nSamples
  let nFeatures = X.nFeatures
  var A = zeros([nSamples, 3])
  result = newSeqWith(nSamples, 0.0)

  linear(X, self.w, result)
  for i in 0..<nSamples:
    result[i] += self.intercept

  if (nFeatures != self.P.shape[1]):
    raise newException(ValueError, "Invalid nFeatures.")
  for s in 0..<self.P.shape[0]:
    if self.ignoreDiag:
      anova(X, self.P, A, 2, s, 0)
    else:
      poly(X, self.P, A, 2, s, 0)
    for i in 0..<nSamples:
      result[i] += self.lams[s]*A[i, 2]


proc dump*(self: ConvexFactorizationMachine, fname: string) =
  ## Dumps the fitted factorization machine.
  self.checkInitialized()
  let nComponents = self.P.shape[0]
  let nFeatures = self.P.shape[1]
  var f: File = open(fname, fmWrite)
  f.writeLine("task: ", $self.task)
  f.writeLine("nFeatures: ", nFeatures)
  f.writeLine("degree: ", 2)
  f.writeLine("nComponents: ", self.P.shape[0])
  f.writeLine("maxComponents: ", self.maxComponents)
  f.writeLine("alpha0: ", self.alpha0)
  f.writeLine("alpha: ", self.alpha)
  f.writeLine("beta: ", self.beta)
  f.writeLine("loss: ", self.loss)
  f.writeLine("fitIntercept: ", self.fitIntercept)
  f.writeLine("fitLinear: ", self.fitLinear)
  f.writeLine("randomState: ", self.randomState)
  var params: seq[float64] = newSeq[float64](nFeatures)
  f.writeLine("P:")
  for s in 0..<nComponents:
    for j in 0..<nFeatures:
      params[j] = self.P[s, j]
    f.writeLine(params.join(" "))
  f.writeLine("w:")
  for j in 0..<nFeatures:
    params[j] = self.w[j]
  f.writeLine(params.join(" "))
  f.writeLine("intercept: ", self.intercept)
  f.close()


proc load*(fm: var ConvexFactorizationMachine, fname: string, warmStart: bool)=
  ## Loads the fitted convex factorization machine.
  new(fm)
  var f: File = open(fname, fmRead)
  var nFeatures: int
  var nComponents: int
  fm.task = parseEnum[TaskKind](f.readLine().split(" ")[1])
  discard parseInt(f.readLine().split(" ")[1], nFeatures, 0)
  discard parseInt(f.readLine().split(" ")[1], nComponents, 0)
  discard parseInt(f.readLine().split(" ")[1], fm.maxComponents, 0)
  discard parseFloat(f.readLine().split(" ")[1], fm.alpha0, 0)
  discard parseFloat(f.readLine().split(" ")[1], fm.alpha, 0)
  discard parseFloat(f.readLine().split(" ")[1], fm.beta, 0)
  fm.loss = parseEnum[LossFunctionKind](f.readLine().split(" ")[1])
  fm.fitIntercept = parseBool(f.readLine().split(" ")[1])
  fm.fitLinear = parseBool(f.readLine().split(" ")[1])
  fm.warmStart = warmStart
  discard parseInt(f.readLine().split(" ")[1], fm.randomState, 0)

  var i = 0
  var val: float64
  fm.P = zeros([nComponents, nFeatures])
  discard f.readLine() # read "P[order]:" and discard it
  for s in 0..<nComponents:
    let line = f.readLine()
    var i = 0
    var val: float64
    for j in 0..<nFeatures:
      i.inc(parseFloat(line, val, i))
      i.inc()
      fm.P[s, j] = val
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
