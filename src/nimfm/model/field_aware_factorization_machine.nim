import ../tensor/tensor, fm_base, ../dataset
import random, sequtils, strutils, parseutils, typetraits


type
  FieldAwareFactorizationMachineObj* = object
    task*: TaskKind         ## regression or classification.
    nComponents*: int       ## Number of basis vectors (rank hyperparameter).
    fitIntercept*: bool     ## Whether to fit intercept (a.k.a bias) term.
    fitLinear*: bool        ## Whether to fit linear term.
    warmStart*: bool        ## Whether to do warwm start fitting.
    randomState*: int       ## The seed of the pseudo random number generator.
    scale*: float64         ## The scale (a.k.a std) of Normal distribution for
                            ## initialization of higher-order weights.
    isInitialized*: bool
    P*: Tensor              ## Weights for the polynomial.
                            ## shape (nFields, nFeatures, nComponents)
    w*: Vector              ## Weigths for linear term, shape: (nFeatures)
    intercept*: float64     ## Intercept term.

  FieldAwareFactorizationMachine* = ref FieldAwareFactorizationMachineObj


proc newFieldAwareFactorizationMachine*(
  task: TaskKind, nComponents = 10, fitIntercept = true, fitLinear = true,
  warmStart = false, randomState = 1, scale = 0.01): FieldAwareFactorizationMachine =
  ## Create a new FactorizationMachine.
  ## task: classification or regression.
  ## nComponents: Number of basis vectors (rank hyperparameter).
  ## fitIntercept: Whether to fit intercept (a.k.a bias) term or not.
  ## fitLinear: Whether to fit linear term or not.
  ## warmStart: Whether to do warwm start fitting or not.
  ## randomState: The seed of the pseudo random number generator.
  ## scale: The scale (a.k.a std) of the Gaussian distribution for initialization
  ##        of higher-order weights.
  new(result)
  result.task = task
  if nComponents < 1:
    raise newException(ValueError, "nComponents < 1.")
  result.nComponents = nComponents
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.warmStart = warmStart
  result.randomState = randomState
  result.scale = scale
  result.isInitialized = false


proc nAugments*(self: FieldAwareFactorizationMachine): int = 0


proc decisionFunction*(self: FieldAwareFactorizationMachine,
                       X: RowFieldDataset): seq[float64] =
  ## Returns the model outputs as seq[float64].
  self.checkInitialized()

  let nSamples: int = X.nSamples
  
  let nFeatures = X.nFeatures
  if (nFeatures != self.P.shape[1]):
    raise newException(ValueError, "Invalid nFeatures.")
  let nFields = X.nFields
  if nFields != self.P.shape[0]:
    raise newException(ValueError, "Invalid nFields.")

  result = newSeqWith(nSamples, self.intercept)

  for i in 0..<nSamples:
    for (j, val) in X.getRow(i):
      result[i] += val * self.w[j]

  for i in 0..<nSamples:
    for (f1, j1, val1) in X.getRowWithField(i):
      for (f2, j2, val2) in X.getRowWithField(i):
        if j1 < j2:
          result[i] += val1 * val2 * dot(self.P[f2, j1], self.P[f1, j2])


proc init*(self: FieldAwareFactorizationMachine, X: RowFieldDataset, force=false) =
  ## Initializes the field-aware factorization machine.
  ## self will not be initialized if force=false, ffm is already initialized,
  ## and warmStart=true.
  if force or not (self.warmStart and self.isInitialized):
    let nFeatures: int = X.nFeatures
    let nFields = X.nFields
    randomize(self.randomState)

    self.w = zeros([nFeatures])
    self.P = randomNormal([nFields, nFeatures, self.nComponents],
                           scale = self.scale)
    self.intercept = 0.0
  self.isInitialized = true


proc dump*(self: FieldAwareFactorizationMachine, fname: string) =
  ## Dumps the fitted factorization machine.
  self.checkInitialized()
  let nFields = self.P.shape[0]
  let nFeatures = self.P.shape[1]
  var f: File = open(fname, fmWrite)
  f.writeLine("task: ", $self.task)
  f.writeLine("nFields: ", nFields)
  f.writeLine("nFeatures: ", nFeatures)
  f.writeLine("nComponents: ", self.nComponents)
  f.writeLine("fitIntercept: ", self.fitIntercept)
  f.writeLine("fitLinear: ", self.fitLinear)
  f.writeLine("randomState: ", self.randomState)
  f.writeLine("scale: ", self.scale)
  for field in 0..<self.P.shape[0]:
    f.writeLine("P[", field, "]:")
    for j in 0..<nFeatures:
      f.writeLine(self.P[field, j].join(" "))
  f.writeLine("w:")
  f.writeLine(self.w.join(" "))
  f.writeLine("intercept: ", self.intercept)
  f.close()


proc load*(ffm: var FieldAwareFactorizationMachine, fname: string, warmStart: bool) =
  ## Loads the fitted factorization machine.
  new(ffm)
  var f: File = open(fname, fmRead)
  var nFeatures, nFields: int
  ffm.task = parseEnum[TaskKind](f.readLine().split(" ")[1])
  discard parseInt(f.readLine().split(" ")[1], nFields, 0)
  discard parseInt(f.readLine().split(" ")[1], nFeatures, 0)
  discard parseInt(f.readLine().split(" ")[1], ffm.nComponents, 0)
  ffm.fitIntercept = parseBool(f.readLine().split(" ")[1])
  ffm.fitLinear = parseBool(f.readLine().split(" ")[1])
  ffm.warmStart = warmStart
  discard parseInt(f.readLine().split(" ")[1], ffm.randomState, 0)
  discard parseFloat(f.readLine().split(" ")[1], ffm.scale, 0)

  var i = 0
  var val: float64

  # read P
  ffm.P = zeros([nFields, nFeatures, ffm.nComponents])
  for field in 0..<nFields:
    discard f.readLine() # read "P[order]:" and discard it
    for j in 0..<nFeatures:
      let line = f.readLine()
      i = 0
      for s in 0..<ffm.nComponents:
        i.inc(parseFloat(line, val, i))
        i.inc()
        ffm.P[field, j, s] = val
  ffm.w = zeros([nFeatures])
  discard f.readLine()
  let line = f.readLine()
  i = 0
  for j in 0..<nFeatures:
    i.inc(parseFloat(line, val, i))
    i.inc()
    ffm.w[j] = val
  discard parseFloat(f.readLine().split(" ")[1], ffm.intercept, 0)
  f.close()
  ffm.isInitialized = true
