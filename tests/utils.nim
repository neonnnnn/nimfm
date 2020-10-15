from nimfm/models/factorization_machine import FitLowerKind
import nimfm/tensor/tensor, nimfm/dataset, nimfm/models/fm_base
import sequtils, random, unittest
import models/fm_slow


proc createFMDataset*(X: var CSCDataset, y: var seq[float64],
                      n, d, degree, nComponents, randomstate: int,
                      fitLower: FitLowerKind = explicit,
                      fitLinear = true, fitIntercept = true,
                      scale=1.0, threshold=0.0) =
  var fm = newFMSlow(
    task = regression, fitLinear = fitLinear, degree = degree,
    nComponents = nComponents, fitLower = fitLower,
    fitIntercept = fitIntercept, randomState = randomState,
    scale=scale)

  var data: seq[seq[float64]] = newSeqWith(n, newSeqWith(d, 0.0))
  for i in 0..<n:
    for j in 0..<d:
      data[i][j] = rand(2.0) - 1.0
      if abs(data[i][j]) < threshold :
        data[i][j] = 0
  X = toCSCDataset(data)
  fm.init(toMatrix(data))
  y = fm.decisionFunction(toMatrix(data))


proc createFMDataset*(X: var CSRDataset, y: var seq[float64],
                      n, d, degree, nComponents, randomstate: int,
                      fitLower: FitLowerKind = explicit,
                      fitLinear = true, fitIntercept = true,
                      scale=1.0, threshold=0.0) =
  var fm = newFMSlow(
    task = regression, fitLinear = fitLinear, degree = degree,
    nComponents = nComponents, fitLower = fitLower,
    fitIntercept = fitIntercept, randomState = randomState,
    scale=scale)

  var data: seq[seq[float64]] = newSeqWith(n, newSeqWith(d, 0.0))
  for i in 0..<n:
    for j in 0..<d:
      data[i][j] = rand(2.0) - 1.0
      if abs(data[i][j]) < threshold: data[i][j] = 0.0
  X = toCSRDataset(data)
  fm.init(toMatrix(data))
  y = fm.decisionFunction(toMatrix(data))


proc checkAlmostEqual*(actual, desired: Tensor, rtol = 1e-6, atol = 1e-9) =
  check actual.shape == desired.shape
  if actual.shape == desired.shape:
    for i in 0..<actual.shape[0]:
      for j in 0..<actual.shape[1]:
        for k in 0..<actual.shape[2]:
          let diff = abs(actual[i, j, k] - desired[i, j, k])
          check diff <= (atol + abs(desired[i, j, k])*rtol)


proc checkAlmostEqual*(actual, desired: Matrix, rtol = 1e-6, atol = 1e-9) =
  check actual.shape == desired.shape
  if actual.shape == desired.shape:
    for i in 0..<actual.shape[0]:
      for j in 0..<actual.shape[1]:
        let diff = abs(actual[i, j] - desired[i, j])
        check diff <= (atol + abs(desired[i, j])*rtol)


proc checkAlmostEqual*(actual, desired: Vector, rtol = 1e-6, atol = 1e-9) =
  check actual.shape == desired.shape
  if actual.shape == desired.shape:
    for i in 0..<actual.shape[0]:
      check abs(actual[i] - desired[i]) <= (atol + abs(desired[i])*rtol)