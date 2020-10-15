import nimfm/tensor/tensor, nimfm/metrics
import nimfm/models/fm_base, nimfm/models/convex_factorization_machine
import sugar, sequtils, math


type
  CFMSlow* = ref ConvexFactorizationMachineObj


proc newCFMSlow*(
  task: TaskKind, maxComponents = 30,
  fitIntercept = true, fitLinear = true,
  ignoreDiag=true, warmStart = false): CFMSlow =
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
  result.lams = zeros([maxComponents])


proc init*(self: CFMSlow, X: Matrix, force=false) =
  ## Initializes the factorization machine.
  ## If force=false, fm is already initialized, and warmStart=true,
  ## fm will not be initialized.
  if force or not (self.warmStart and self.isInitialized):
    let nFeatures: int = X.shape[1]
    self.w = zeros([nFeatures])
    self.P = zeros([self.maxComponents, nFeatures])
    self.intercept = 0.0
  self.isInitialized = true


proc decisionFunction*(self: CFMSlow, X: Matrix): seq[float64] =
  self.checkInitialized()
  let nSamples: int = X.shape[0]
  let nFeatures = X.shape[1]
  result = newSeqWith(nSamples, self.intercept)

  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      result[i] += self.w[j] * X[i, j]
  
  if nFeatures != self.P.shape[1]:
    raise newException(ValueError, "Invalid nFeatures.")

  for i in 0..<nSamples:
    for s in 0..<len(self.lams):
      var kernel = 0.0
      if self.ignoreDiag:
        for j1 in 0..<nFeatures:
          for j2 in j1+1..<nFeatures:
            kernel += X[i, j1] * X[i, j2] * self.P[s, j1] * self.P[s, j2]
      else:
        for j in 0..<nFeatures:
          kernel += self.P[s, j] * X[i, j]
        kernel = kernel * kernel
      result[i] += self.lams[s] * kernel


proc predict*(self: CFMSlow, X: Matrix): seq[int] =
  result = decisionFunction(self, X).map(x=>sgn(x))


proc checkTarget*(self: CFMSlow, y: seq[SomeNumber]): seq[float64] =
  case self.task
  of classification:
    result = y.map(x => float(sgn(x)))
  of regression:
    result = y.map(x => float(x))


proc score*(self: CFMSlow, X: Matrix, y: seq[float64]): float64 =
  let yPred = self.decisionFunction(X)
  case self.task
  of regression:
    result = rmse(y, yPred)
  of classification:
    result = accuracy(y.map(x=>sgn(x)), yPred.map(x=>sgn(x)))