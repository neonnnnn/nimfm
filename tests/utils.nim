import nimfm/factorization_machine, nimfm/tensor, nimfm/dataset
import sequtils, random, unittest


proc createFMDataset*(X: var CSCDataset, y: var seq[float64],
                      n, d, degree, nComponents, randomstate: int,
                      fitLower: FitLowerKind,
                      fitLinear, fitIntercept: bool) =
  var fm = newFactorizationMachine(
    task = regression, fitLinear = fitLinear, degree = degree,
    nComponents = nComponents, fitLower = fitLower,
    fitIntercept = fitIntercept, randomState = randomState)


  var data: seq[seq[float64]] = newSeqWith(n, newSeqWith(d, 0.0))
  for i in 0..<n:
    for j in 0..<d:
      data[i][j] = random(1.0)
  X = toCSC(data)
  fm.init(X)
  y = fm.decisionFunction(X)


proc checkAlmostEqual*(actual, desired: Tensor, rtol = 1e-3, atol = 1e-3) =
  check actual.shape == desired.shape
  if actual.shape == desired.shape:
    for i in 0..<actual.shape[0]:
      for j in 0..<actual.shape[1]:
        for k in 0..<actual.shape[2]:
          let diff = abs(actual[i, j, k] - desired[i, j, k])
          check diff <= (atol + abs(desired[i, j, k])*rtol)


proc checkAlmostEqual*(actual, desired: Matrix, rtol = 1e-3, atol = 1e-3) =
  check actual.shape == desired.shape
  if actual.shape == desired.shape:
    for i in 0..<actual.shape[0]:
      for j in 0..<actual.shape[1]:
        let diff = abs(actual[i, j] - desired[i, j])
        check diff <= (atol + abs(desired[i, j])*rtol)



proc checkAlmostEqual*(actual, desired: Vector, rtol = 1e-3, atol = 1e-3) =
  check actual.shape == desired.shape
  if actual.shape == desired.shape:
    for i in 0..<actual.shape[0]:
      check abs(actual[i] - desired[i]) <= (atol + abs(desired[i])*rtol)
