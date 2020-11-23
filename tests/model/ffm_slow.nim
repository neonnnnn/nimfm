import nimfm/tensor/tensor, nimfm/metrics
import nimfm/model/fm_base
from nimfm/model/field_aware_factorization_machine import 
  FieldAwareFactorizationMachineObj
import sugar, random, sequtils, math


type
  FFMSlow* = ref FieldAwareFactorizationMachineObj

  NotFittedError = object of Exception


proc checkInitialized*(self: FFMSlow) =
  if not self.isInitialized:
    raise newException(NotFittedError, "Factorization machines is not fitted.")


proc nAugments*(self: FFMSlow): int = 0


proc newFFMSlow*(task: TaskKind, n_components = 10, fitIntercept = true, 
                 fitLinear = true, warmStart = false, randomState = 1,
                 scale = 0.01): FFMSlow =
  new(result)
  result.task = task
  if n_components < 1:
    raise newException(ValueError, "nComponents < 1.")
  result.n_components = n_components
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.warmStart = warmStart
  result.randomState = randomState
  result.scale = scale
  result.isInitialized = false


proc decisionFunction*(self: FFMSlow, X: Matrix, fields: seq[int],
                       i: int): float64 =
  let nFeatures = X.shape[1]
  let nFields = max(fields) + 1
  let nAugments = self.nAugments

  result = self.intercept
  for j in 0..<nFeatures:
    result += self.w[j] * X[i, j]
  
  for j1 in 0..<(nFeatures+nAugments):
    let f1 = if j1 < nFeatures: fields[j1] else: nFields
    let val1 = if j1 < nFeatures: X[i, j1] else: 1.0
    for j2 in (j1+1)..<(nFeatures+nAugments):
      let f2 = if j2 < nFeatures: fields[j2] else: nFields
      let val2 = if j2 < nFeatures: X[i, j2] else: 1.0
      let interaction = val1 * val2
      for s in 0..<self.nComponents:
        result += interaction * self.P[f2, j1, s] * self.P[f1, j2, s]
    

proc decisionFunction*(self: FFMSlow, X: Matrix, fields: seq[int]): seq[float64] =
  self.checkInitialized()
  let nSamples: int = X.shape[0]
  let nFeatures = X.shape[1]
  let nAugments = self.nAugments
  result = newSeqWith(nSamples, 0.0)
  if (nAugments + nFeatures != self.P.shape[1]):
    raise newException(ValueError, "Invalid nFeatures.")
  for i in 0..<nSamples:
    result[i] = decisionFunction(self, X, fields, i)
  

proc predict*(self: FFMSlow, X: Matrix, fields: seq[int]): seq[int] =
  result = decisionFunction(self, X, fields).map(x=>sgn(x))


proc init*(self: FFMSlow, X: Matrix, fields: seq[int]) =
  if not (self.warmStart and self.isInitialized):
    let nFeatures: int = X.shape[1]
    randomize(self.randomState)
    let nFields = max(fields) + 1
    let nAugments = self.nAugments
    self.w = zeros([nFeatures])
    if nAugments > 0:
      self.P = randomNormal([nFields + 1, nFeatures+nAugments, self.nComponents],
                             scale = self.scale)
    else:
      self.P = randomNormal([nFields, nFeatures+nAugments, self.nComponents],
                             scale = self.scale)

    self.intercept = 0.0
  self.isInitialized = true


proc checkTarget*(self: FFMSlow, y: seq[SomeNumber]): seq[float64] =
  case self.task
  of classification:
    result = y.map(x => float(sgn(x)))
  of regression:
    result = y.map(x => float(x))


proc score*(self: FFMSlow, X: Matrix, fields: seq[int], y: seq[float64]): float64 =
  let yPred = self.decisionFunction(X, fields)
  case self.task
  of regression:
    result = rmse(y, yPred)
  of classification:
    result = accuracy(y.map(x=>sgn(x)), yPred.map(x=>sgn(x)))


proc computeGrad*(self: FFMSlow, X: Matrix, fields: seq[int], i: int,
                  dL: float64, grad: var Tensor) =
  let
    nFeatures = X.shape[1]
    nAugments = self.nAugments
    nComponents = self.nComponents
    nFields = max(fields)+1

  for j1 in 0..<(nFeatures+nAugments):
    let f1 = if j1 < nFeatures: fields[j1] else: nFields
    let val1 = if j1 < nFeatures: X[i, j1] else: 1.0
    for j2 in (j1+1)..<(nFeatures+nAugments):
      let f2 = if j2 < nFeatures: fields[j2] else: nFields
      let val2 = if j2 < nFeatures: X[i, j2] else: 1.0
      let interaction = val1 * val2
      for s in 0..<nComponents:
        grad[f2, j1, s] += dL * self.P[f1, j2, s] * interaction
        grad[f1, j2, s] += dL * self.P[f2, j1, s] * interaction
