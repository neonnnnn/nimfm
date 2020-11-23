import sequtils, math, sugar
import ../metrics, ../utils


type
  TaskKind* = enum
    regression = "r",
    classification = "c"

  NotFittedError = object of Exception


proc checkInitialized*[FM](self: FM) =
  if not self.isInitialized:
    raise newException(NotFittedError, "Factorization machines is not fitted.")


proc predict*[Dataset, FM](self: FM, X: Dataset): seq[int] =
  ## Returns the sign vector of the model outputs as seq[int].
  result = self.decisionFunction(X).map(x=>sgn(x))


proc predictProba*[Dataset, FM](self: FM, X: Dataset): seq[float64] =
  ## Returns probabilities that each instance belongs to positive class.
  ## It shoud be used only when task=classification.
  result = self.decisionFunction(X).map(expit)


proc checkTarget*[FM](self: FM, y: seq[SomeNumber]): seq[float64] =
  ## Transforms targets vector to float for regression or
  ## to sign for classification.
  case self.task
  of classification:
    result = y.map(x => float(sgn(x)))
  of regression:
    result = y.map(x => float(x))


proc score*[FM, Dataset](self: FM, X: Dataset, y: seq[float64]): float64 =
  ## Returns the score between the model outputs and true targets.
  ## Computes root mean squared error when task=regression (lower is better).
  ## Computes accuracy when task=classification (higher is better). 
  let yPred = self.decisionFunction(X)
  case self.task
  of regression:
    result = rmse(y, yPred)
  of classification:
    result = accuracy(y.map(x=>sgn(x)), yPred.map(x=>sgn(x)))