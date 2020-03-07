import nimfm/factorization_machine, nimfm/tensor, nimfm/dataset
import sequtils, random, unittest


proc createFMDataset*(X: var CSCDataset, y: var seq[float64],
                      n, d, degree, nComponents, randomstate: int,
                      fitLower: FitLowerKind,
                      fitLinear, fitIntercept: bool,
                      scale=0.1, threshold=0.0) =
  var fm = newFactorizationMachine(
    task = regression, fitLinear = fitLinear, degree = degree,
    nComponents = nComponents, fitLower = fitLower,
    fitIntercept = fitIntercept, randomState = randomState,
    scale=scale)

  var data: seq[seq[float64]] = newSeqWith(n, newSeqWith(d, 0.0))
  for i in 0..<n:
    for j in 0..<d:
      data[i][j] = rand(1.0)
      if abs(data[i][j]) < threshold :
        data[i][j] = 0
  X = toCSC(data)
  fm.init(X)
  y = fm.decisionFunction(X)


proc createFMDataset*(X: var CSRDataset, y: var seq[float64],
                      n, d, degree, nComponents, randomstate: int,
                      fitLower: FitLowerKind,
                      fitLinear, fitIntercept: bool,
                      scale=0.1, threshold=0.0) =
  var fm = newFactorizationMachine(
    task = regression, fitLinear = fitLinear, degree = degree,
    nComponents = nComponents, fitLower = fitLower,
    fitIntercept = fitIntercept, randomState = randomState,
    scale=scale)


  var data: seq[seq[float64]] = newSeqWith(n, newSeqWith(d, 0.0))
  for i in 0..<n:
    for j in 0..<d:
      data[i][j] = rand(1.0)
      if abs(data[i][j]) < threshold: data[i][j] = 0.0
  X = toCSR(data)
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


proc comb*(n, m: int, k=0): seq[seq[int]] =
  result = @[]
  if m == 1:
    for i in k..<n:
      result.add(@[i])
  else:
    for i in k..<(n-m+1):
      for val in comb(n, m-1, i+1):
        result.add(@[i] & val)


proc combNotj*(n, m, j: int, k=0): seq[seq[int]] =
  result = @[]
  if m == 1:
    for i in k..<n:
      if i != j:
        result.add(@[i])
  else:
    for i in k..<(n-m+1):
      if i != j:
        for val in combNotj(n, m-1, j, i+1):
          result.add(@[i] & val)