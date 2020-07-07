import strformat, math
import tensor, dataset


proc matmul*(D: Matrix, S: CSRDataset, R: var Matrix) =
  if D.shape[1] != S.nSamples:
    let msg = fmt"D.shape[1] {D.shape[1]} != nSamples {S.nSamples}."
    raise newException(ValueError, msg)

  R[0..^1, 0..^1] = 0.0
  for m in 0..<D.shape[0]:
    for i in 0..<S.nSamples:
      for (j, val) in S.getRow(i):
        R[m, j] += D[m, i] * val
      

proc matmul*(D: Matrix, S: CSCDataset, R: var Matrix) =
  if D.shape[1] != S.nSamples:
    let msg = fmt"D.shape[1] {D.shape[1]} != nSamples {S.nSamples}."
    raise newException(ValueError, msg)

  R[0..^1, 0..^1] = 0.0
  for m in 0..<D.shape[0]:
    for j in 0..<S.nFeatures:
      for (i, val) in S.getCol(j):
        R[m, j] += D[m, i] * val
 

proc matmul*(S: CSRDataset, D: Matrix, R: var Matrix) =
  if D.shape[0] != S.nSamples:
    let msg = fmt"nFeatures {S.nFeatures} != D.shape[0] {D.shape[0]}."
    raise newException(ValueError, msg) 
  R[0..^1, 0..^1] = 0.0

  for i in 0..<S.nSamples:
    for (j, val) in S.getRow(i):
      for n in 0..<D.shape[1]:
        R[i, n] += val * D[j, n]


proc matmul*(S: CSCDataset, D: Matrix, R: var Matrix) =
  if D.shape[0] != S.nSamples:
    let msg = fmt"nFeatures {S.nFeatures} != D.shape[0] {D.shape[0]}."
    raise newException(ValueError, msg) 

  R[0..^1, 0..^1] = 0.0
  for j in 0..<S.nFeatures:
    for (i, val) in S.getCol(j):
      for n in 0..<D.shape[1]:
        R[i, n] += val * D[j, n]


proc matmul*[T: CSRDataset|CSCDataset](D: Matrix, S: T): Matrix =
  new(result)
  result = zeros([D.shape[0], S.nFeatures])
  matmul(D, S, result)


proc matmul*[T: CSRDataset|CSCDataset](S: T, D: Matrix): Matrix =
  new(result)
  result = zeros([S.nSamples, D.shape[1]])
  matmul(S, D, result)


proc mvmul*(S: CSRDataset, vec: Vector, result: var Vector) =
  if S.nFeatures != len(vec):
    let msg = fmt"nFeatures {S.nFeatures} != len(vec){len(vec)}."
    raise newException(ValueError, msg)

  for i in 0..<S.nSamples:
    result[i] = 0.0
    for (j, val) in S.getRow(i):
      result[i] += val * vec[j]


proc mvmul*(S: CSCDataset, vec: Vector, result: var Vector) =
  if S.nFeatures != len(vec):
    let msg = fmt"nFeatures {S.nFeatures} != len(vec){len(vec)}."
    raise newException(ValueError, msg)

  result[0..^1] = 0.0
  for j in 0..<S.nFeatures:
    for (i, val) in S.getCol(j):
      result[i] += val * vec[j]


proc vmmul*(vec: Vector, S: CSRDataset, result: var Vector) =
  if S.nSamples != len(vec):
    let msg = fmt"len(vec){len(vec)} != nSamples {S.nSamples}."
    raise newException(ValueError, msg)

  result[0..^1] = 0.0
  for i in 0..<S.nSamples:
    for (j, val) in S.getRow(i):
      result[j] += val * vec[i]


proc vmmul*(vec: Vector, S: CSCDataset, result: var Vector) =
  if S.nSamples != len(vec):
    let msg = fmt"len(vec){len(vec)} != nSamples {S.nSamples}."
    raise newException(ValueError, msg)

  for j in 0..<S.nFeatures:
    result[j] = 0
    for (i, val) in S.getCol(j):
      result[j] += val * vec[i]


proc mvmul*[T: CSRDataset|CSCDataset](S: T, vec: Vector): Vector =
  result = zeros([S.nSamples])
  mvmul(S, vec, result)


proc vmmul*[T: CSRDataset|CSCDataset](vec: Vector, S: T): Vector =
  result = zeros([S.nFeatures])
  vmmul(vec, S, result)


proc norm*(X: CSRDataset, p=1, axis=0): Vector =
  if axis == 0:
    result = zeros([X.nFeatures])
    for i in 0..<X.nSamples:
      for (j, val) in X.getRow(i):
        result[j] += val ^ p
  elif axis == 1:
    result = zeros([X.nSamples])
    for i in 0..<X.nSamples:
      for (j, val) in X.getRow(i):
        result[i] += val ^ p
  for i in 0..<len(result):
    result[i] = pow(result[i], 1.0 / float(p))


proc norm*(X: CSCDataset, p=1, axis=0): Vector =
  if axis == 0:
    result = zeros([X.nFeatures])
    for j in 0..<X.nFeatures:
      for (i, val) in X.getCol(j):
        result[j] += val ^ p
  elif axis == 1:
    result = zeros([X.nSamples])
    for j in 0..<X.nFeatures:
      for (i, val) in X.getCol(j):
        result[i] += val ^ p
  for i in 0..<len(result):
    result[i] = pow(result[i], 1.0 / float(p))
  