import nimfm/loss, nimfm/tensor, nimfm/metrics, nimfm/factorization_machine
import nimfm/fm_base
import sugar, random, sequtils, math
import utils

export nAugments


type
  FMSlow*[L] = ref FactorizationMachineObj[L]

  NotFittedError = object of Exception


proc checkInitialized*(self: FMSlow) =
  if not self.isInitalized:
    raise newException(NotFittedError, "Factorization machines is not fitted.")


proc newFMSlow*[L](task: TaskKind, degree = 2, n_components = 30, alpha0 = 1e-6,
                   alpha = 1e-3, beta = 1e-3, loss: L = newSquared(),
                   fitLower = explicit, fitIntercept = true, fitLinear = true,
                   warmStart = false, randomState = 1, scale = 0.01): FMSlow[L] =
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
  result.loss = loss
  result.fitLower = fitLower
  result.fitIntercept = fitIntercept
  result.fitLinear = fitLinear
  result.warmStart = warmStart
  result.randomState = randomState
  result.scale = scale
  result.isInitalized = false


proc decisionFunction*[L](self: FMSlow[L], X: Matrix): seq[float64] =
  self.checkInitialized()
  let nSamples: int = X.shape[0]
  let nComponents: int = self.nComponents
  let nFeatures = X.shape[1]
  result = newSeqWith(nSamples, self.intercept)

  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      result[i] += self.w[j] * X[i, j]
  
  var nAugments = self.nAugments
  if (nAugments + nFeatures != self.P.shape[2]):
    raise newException(ValueError, "Invalid nFeatures.")
  for i in 0..<nSamples:
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
        result[i] += anova


proc predict*[L](self: FMSlow[L], X: Matrix): seq[int] =
  result = decisionFunction(self, X).map(x=>sgn(x))


proc init*[L](self: FMSlow[L], X: Matrix) =
  if not (self.warmStart and self.isInitalized):
    let nFeatures: int = X.shape[1]
    randomize(self.randomState)

    self.w = zeros([nFeatures])
    let nOrders = self.nOrders
    let nAugments = self.nAugments
    self.P = randomNormal([nOrders, self.nComponents, nFeatures+nAugments],
                           scale = self.scale)
    self.intercept = 0.0
  self.isInitalized = true


proc checkTarget*[L](self: FMSlow[L], y: seq[SomeNumber]): seq[float64] =
  case self.task
  of classification:
    result = y.map(x => float(sgn(x)))
  of regression:
    result = y.map(x => float(x))


proc score*[L](self: FMSlow[L], X: Matrix, y: seq[float64]): float64 =
  let yPred = self.decisionFunction(X)
  case self.task
  of regression:
    result = rmse(y, yPred)
  of classification:
    result = accuracy(y.map(x=>sgn(x)), yPred.map(x=>sgn(x)))