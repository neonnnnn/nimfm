import math, sequtils, algorithm, sugar
import utils


proc rmse*(yTrue, yScore: seq[float64]): float64 =
  ## Returns root mean squared error.
  if len(yTrue) != len(yScore):
    let msg = "len(yScore)=" & $len(yScore) & ", but len(yTrue)=" & $len(yTrue)
    raise newException(ValueError, msg)
  result = 0.0
  for (val1, val2) in zip(yScore, yTrue):
    result += pow(val1-val2, 2)
  result = sqrt(result / float(len(yTrue)))


proc r2*(yTrue, yScore: seq[float64]): float64 =
  ## Returns r2 score (the coefficient of determination).
  if len(yTrue) != len(yScore):
    let msg = "len(yScore)=" & $len(yScore) & ", but len(yTrue)=" & $len(yTrue)
    raise newException(ValueError, msg)

  let nSamples = yTrue.len
  var res = 0.0
  for (target, score) in zip(yTrue, yScore):
    res += (target-score)^2

  if res == 0:
    result = 1.0
  else:
    let mean = sum(yTrue) / float(nSamples)
    let tot = sum(yTrue.map(x=>(x-mean)^2))
    if tot != 0.0:
      result = 1.0 - res / tot
    else:
      echo("All instances have same target value.")
      result = 0.0


proc accuracy*(yTrue, yPred: seq[int]): float64 =
  ## Returns accuracy.
  if len(yPred) != len(yTrue):
    let msg = "len(yPred)=" & $len(yPred) & ", but len(yTrue)=" & $len(yTrue)
    raise newException(ValueError, msg)
  result = 0.0
  for (val1, val2) in zip(yPred, yTrue):
    result += float(val1 == val2)
  result /= float(len(yPred))


proc precisionRecallFscore*(yTrue, yPred: seq[int], pos=1):
                            tuple[prec, recall, fscore: float64] =
  ## Returns precision, recall, and F1-score for "binary classification".
  var
    tp, fp, tn, fn: float64
  if len(yPred) != len(yTrue):
    let msg = "len(yPred)=" & $len(yPred) & ", but len(yTrue)=" & $len(yTrue)
    raise newException(ValueError, msg)
  
  let nUnique = len(deduplicate(yTrue))
  if nUnique > 2:
    echo("yTrue has " & $nUnique & " unique values. " & 
         "All values that are not " & $pos & " are regarded as.")
  for (target, pred) in zip(yTrue, yPred):
    if target == pos:
      if pred == pos: tp += 1.0
      else: fn += 1.0
    else:
      if pred == pos: fp += 1.0
      else: tn += 1.0
  let prec = if (tp+fp) != 0: tp / (tp+fp) else: 0.0
  let recall = if (tp+fn) != 0: tp / (tp+fn) else: 0.0
  let fscore = if (prec+recall) != 0: 2*prec*recall/(prec+recall) else: 0
  result = (prec, recall, fscore)


proc rocauc*(yTrue: seq[int], yScore: seq[float64], pos:int = 1): float64 =
  ## Returns the area under the receiver operating characteristic curve
  ## for "binary classification".
  let indicesSorted = argsort(yScore, SortOrder.Descending)
  result = 0.0
  if len(yTrue) != len(yScore):
    let msg = "len(yScore)=" & $len(yScore) & ", but len(yTrue)=" & $len(yTrue)
    raise newException(ValueError, msg)
  var
    fp, tp, fpPrev, tpPrev: int
    scorePrev: float64 = NegInf
    np, nn: int
  for i in indicesSorted:
    if yScore[i] != scorePrev:
      result += float((fp - fpPrev) * (tp + tpPrev)) / 2.0
      scorePrev = yScore[i]
      fpPrev = fp
      tpPrev = tp
    
    if yTrue[i] == pos: 
      np += 1
      tp += 1
    else:
      nn += 1
      fp += 1

  result += float((fp - fpPrev) * (tp + tpPrev)) / 2.0
  result /= float(nn*np)
