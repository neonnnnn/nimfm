import math, sequtils

proc rmse*(yPred, y: seq[float64]): float64 =
  if len(yPred) != len(y):
    let msg = "len(yPred)=" & $len(yPred) & ", but len(y)=" & $len(y)
    raise newException(ValueError, msg)
  result = 0.0
  for (val1, val2) in zip(yPred, y):
    result += pow(val1-val2, 2)
  result = sqrt(result / float(len(yPred)))


proc accuracy*(yPred, y: seq[int]): float64 =
  if len(yPred) != len(y):
    let msg = "len(yPred)=" & $len(yPred) & ", but len(y)=" & $len(y)
    raise newException(ValueError, msg)
  result = 0.0
  for (val1, val2) in zip(yPred, y):
    result += float(val1 == val2)
  result /= float(len(yPred))
