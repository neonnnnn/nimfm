import nimfm/tensor/tensor, nimfm/metrics
import nimfm/model/fm_base
from nimfm/model/factorization_machine 
  import FactorizationMachineObj, FitLowerKind, nAugments, nOrders
import sugar, random, sequtils, math
import ../comb

export nAugments


type
  FMSlow* = ref FactorizationMachineObj

  NotFittedError = object of Exception


proc checkInitialized*(self: FMSlow) =
  if not self.isInitialized:
    raise newException(NotFittedError, "Factorization machines is not fitted.")


proc newFMSlow*(task: TaskKind, degree = 2, n_components = 30,
                fitLower = explicit, fitIntercept = true, fitLinear = true,
                warmStart = false, randomState = 1, scale = 0.01): FMSlow =
  new(result)
  result.task = task
  if degree < 1:
    raise newException(ValueError, "degree < 1.")
  result.degree = degree
  if n_components < 1:
    raise newException(ValueError, "nComponents < 1.")
  result.n_components = n_components
  result.fitLower = fitLower
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.warmStart = warmStart
  result.randomState = randomState
  result.scale = scale
  result.isInitialized = false


proc decisionFunction*(self: FMSlow, X: Matrix, i: int): float64 =
  let
    nFeatures = X.shape[1]
    nComponents = self.nComponents
    nAugments = self.nAugments
  result = self.intercept
  for j in 0..<nFeatures:
    result += self.w[j] * X[i, j]
  for order in 0..<self.nOrders:
    for s in 0..<nComponents:
      var anova = 0.0
      for indices in comb(nFeatures+nAugments, self.degree-order):
        var prod = 1.0
        for j in indices:
          prod *= self.P[order, s, j]
          if j < nFeatures:
            prod *= X[i, j]
        anova += prod
      result += anova


proc decisionFunction*(self: FMSlow, X: Matrix): seq[float64] =
  self.checkInitialized()
  let nSamples: int = X.shape[0]
  let nFeatures = X.shape[1]
  result = newSeqWith(nSamples, 0.0)
  let nAugments = self.nAugments
  if (nAugments + nFeatures != self.P.shape[2]):
    raise newException(ValueError, "Invalid nFeatures.")
  
  for i in 0..<nSamples:
    result[i] = self.decisionFunction(X, i)
      
   
proc predict*(self: FMSlow, X: Matrix): seq[int] =
  result = decisionFunction(self, X).map(x=>sgn(x))


proc init*(self: FMSlow, X: Matrix) =
  if not (self.warmStart and self.isInitialized):
    let nFeatures: int = X.shape[1]
    randomize(self.randomState)

    self.w = zeros([nFeatures])
    let nOrders = self.nOrders
    let nAugments = self.nAugments
    self.P = randomNormal([nOrders, self.nComponents, nFeatures+nAugments],
                           scale = self.scale)
    self.intercept = 0.0
  self.isInitialized = true


proc checkTarget*(self: FMSlow, y: seq[SomeNumber]): seq[float64] =
  case self.task
  of classification:
    result = y.map(x => float(sgn(x)))
  of regression:
    result = y.map(x => float(x))


proc score*(self: FMSlow, X: Matrix, y: seq[float64]): float64 =
  let yPred = self.decisionFunction(X)
  case self.task
  of regression:
    result = rmse(y, yPred)
  of classification:
    result = accuracy(y.map(x=>sgn(x)), yPred.map(x=>sgn(x)))


proc computeGrad*(self: FMSlow, X: Matrix, i: int, dL: float64,
                  grad: var Tensor) =
  let
    nFeatures = X.shape[1]
    nComponents = self.P.shape[1]
    nAugments = self.nAugments

  for order in 0..<self.P.shape[0]:
    for s in 0..<nComponents:
      for j in 0..<(nFeatures+nAugments):
        var tmp = 0.0
        
        for indices in combNotj(nFeatures+nAugments, self.degree-order-1, j):
          var prod = 1.0
          for j2 in indices:
            prod *= self.P[order, s, j2]
            if j2 < nFeatures:
              prod *= X[i, j2]
          tmp += prod
        
        if j < nFeatures:
          tmp *= X[i, j]

        grad[order, s, j] += dL * tmp
