import ../tensor/tensor, ../kernels, fm_base
import strutils, parseutils, sequtils, algorithm, typetraits


type
  ConvexFactorizationMachineObj* = object
    task*: TaskKind         ## regression or classification.
    degree*: int            ## Degree of the polynomial, 2.
    maxComponents*: int     ## Maximum number of basis vectors.
    fitIntercept*: bool     ## Whether to fit intercept (a.k.a bias) term.
    fitLinear*: bool        ## Whether to fit linear term.
    ignoreDiag*: bool       ## Whether ignored diag (FM) or not (PN).
    warmStart*: bool        ## Whether to do warwm start fitting.
    isInitialized*: bool
    P*: Matrix              ## Weights for the polynomial.
                            ## shape (nComponents, nFeatures)
    lams*: Vector           ## Weight for vectors in basis.
                            ## shape: (nComponents)
    w*: Vector              ## Weigths for linear term, shape: (nFeatures)
    intercept*: float64     ## Intercept term.

  ConvexFactorizationMachine* = ref ConvexFactorizationMachineObj


proc newConvexFactorizationMachine*(
  task: TaskKind, maxComponents = 30, fitIntercept = true, fitLinear = true,
  ignoreDiag=true, warmStart = false): ConvexFactorizationMachine =
  ## Create a new ConvexFactorizationMachine.
  ## task: classification or regression.
  ## maxComponents: Maximum number of basis vectors.
  ## fitIntercept: Whether to fit intercept (a.k.a bias) term or not.
  ## fitLinear: Whether to fit linear term or not.
  ## warmStart: Whether to do warwm start fitting or not.
  new(result)
  result.task = task
  result.degree = 2
  if maxComponents < 1:
    raise newException(ValueError, "maxComponents < 1.")
  result.maxComponents = maxComponents
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.ignoreDiag = ignoreDiag
  result.warmStart = warmStart
  result.degree = 2
  result.isInitialized = false
  result.lams = zeros([0])


proc init*[Dataset](self: ConvexFactorizationMachine, X: Dataset, 
                    force=false) =
  ## Initializes the factorization machine.
  ## self will not be initialized if force=false, self is already initialized,
  ## and warmStart=true.
  if force or not (self.warmStart and self.isInitialized):
    let nFeatures: int = X.nFeatures
    self.w = zeros([nFeatures])
    self.P = zeros([0, nFeatures])
    self.lams = zeros([0])
    self.intercept = 0.0
  self.isInitialized = true


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
      anova(X, self.P, A, 2, s)
    else:
      poly(X, self.P, A, 2, s)
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
  f.writeLine("fitIntercept: ", self.fitIntercept)
  f.writeLine("fitLinear: ", self.fitLinear)
  f.writeLine("lams:")
  f.writeLine(self.lams.join(" "))
  f.writeLine("P:")
  for s in 0..<nComponents:
    f.writeLine(self.P[s].join(" "))
  f.writeLine("w:")
  f.writeLine(self.w.join(" "))
  f.writeLine("intercept: ", self.intercept)
  f.close()


proc load*(fm: var ConvexFactorizationMachine, fname: string,
           warmStart: bool) =
  ## Loads the fitted convex factorization machine.
  new(fm)
  var f: File = open(fname, fmRead)
  var nFeatures: int
  var nComponents: int
  var degree: int
  fm.task = parseEnum[TaskKind](f.readLine().split(" ")[1])
  discard parseInt(f.readLine().split(" ")[1], nFeatures, 0)
  discard parseInt(f.readLine().split(" ")[1], degree, 0)
  discard parseInt(f.readLine().split(" ")[1], nComponents, 0)
  discard parseInt(f.readLine().split(" ")[1], fm.maxComponents, 0)
  fm.fitIntercept = parseBool(f.readLine().split(" ")[1])
  fm.fitLinear = parseBool(f.readLine().split(" ")[1])
  fm.warmStart = warmStart

  var i = 0
  var val: float64

  # read lams
  discard f.readLine() # read "lams:"
  let line_lams = f.readLine()
  i = 0
  fm.lams = zeros([nComponents])

  for s in 0..<nComponents:
    i.inc(parseFloat(line_lams, val, i))
    i.inc()
    fm.lams[s] = val
  # read P
  fm.P = zeros([nComponents, nFeatures])
  discard f.readLine() # read "P[order]:" and discard it
  for s in 0..<nComponents:
    let line = f.readLine()
    i = 0
    for j in 0..<nFeatures:
      i.inc(parseFloat(line, val, i))
      i.inc()
      fm.P[s, j] = val
  fm.w = zeros([nFeatures])

  # read w
  discard f.readLine()
  let line_w = f.readLine()
  i = 0
  for j in 0..<nFeatures:
    i.inc(parseFloat(line_w, val, i))
    i.inc()
    fm.w[j] = val
  # read intercept
  discard parseFloat(f.readLine().split(" ")[1], fm.intercept, 0)
  f.close()
  fm.isInitialized = true
