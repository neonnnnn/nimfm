import strformat, math
import tensor/tensor, tensor/sparse, tensor/sparse_stream, dataset

type
  RowData = CSRMatrix|StreamCSRMatrix|CSRDataset|StreamCSRDataset

  ColData = CSCMatrix|StreamCSCMatrix|CSCDataset|StreamCSCDataset


proc matmul*[T: RowData](D: Matrix, S: T, R: var Matrix) =
  if D.shape[1] != S.shape[0]:
    let msg = fmt"D.shape[1] {D.shape[1]} != shape[0] {S.shape[0]}."
    raise newException(ValueError, msg)

  R[0..^1, 0..^1] = 0.0
  for m in 0..<D.shape[0]:
    for i in 0..<S.shape[0]:
      for (j, val) in S.getRow(i):
        R[m, j] += D[m, i] * val
      

proc matmul*[T: ColData](D: Matrix, S: T, R: var Matrix) =
  if D.shape[1] != S.shape[0]:
    let msg = fmt"D.shape[1] {D.shape[1]} != shape[0] {S.shape[0]}."
    raise newException(ValueError, msg)

  R[0..^1, 0..^1] = 0.0
  for m in 0..<D.shape[0]:
    for j in 0..<S.shape[1]:
      for (i, val) in S.getCol(j):
        R[m, j] += D[m, i] * val
 

proc matmul*[T: RowData](S: T, D: Matrix, R: var Matrix) =
  if D.shape[0] != S.shape[0]:
    let msg = fmt"shape[1] {S.shape[1]} != D.shape[0] {D.shape[0]}."
    raise newException(ValueError, msg) 
  R[0..^1, 0..^1] = 0.0

  for i in 0..<S.shape[0]:
    for (j, val) in S.getRow(i):
      for n in 0..<D.shape[1]:
        R[i, n] += val * D[j, n]


proc matmul*[T: ColData](S: T, D: Matrix, R: var Matrix) =
  if D.shape[0] != S.shape[0]:
    let msg = fmt"shape[1] {S.shape[1]} != D.shape[0] {D.shape[0]}."
    raise newException(ValueError, msg) 

  R[0..^1, 0..^1] = 0.0
  for j in 0..<S.shape[1]:
    for (i, val) in S.getCol(j):
      for n in 0..<D.shape[1]:
        R[i, n] += val * D[j, n]


proc matmul*[T: CSRDataset|CSCDataset|CSRMatrix|CSCMatrix](D: Matrix, S: T): Matrix =
  new(result)
  result = zeros([D.shape[0], S.shape[1]])
  matmul(D, S, result)


proc matmul*[T: CSRDataset|CSCDataset|CSRMatrix|CSCMatrix](S: T, D: Matrix): Matrix =
  new(result)
  result = zeros([S.shape[0], D.shape[1]])
  matmul(S, D, result)


proc mvmul*[T: RowData](S: T, vec: Vector, result: var Vector) =
  if S.shape[1] != len(vec):
    let msg = fmt"shape[1] {S.shape[1]} != len(vec){len(vec)}."
    raise newException(ValueError, msg)

  for i in 0..<S.shape[0]:
    result[i] = 0.0
    for (j, val) in S.getRow(i):
      result[i] += val * vec[j]


proc mvmul*[T: ColData](S: T, vec: Vector, result: var Vector) =
  if S.shape[1] != len(vec):
    let msg = fmt"shape[1] {S.shape[1]} != len(vec){len(vec)}."
    raise newException(ValueError, msg)

  result[0..^1] = 0.0
  for j in 0..<S.shape[1]:
    for (i, val) in S.getCol(j):
      result[i] += val * vec[j]


proc vmmul*[T: RowData](vec: Vector, S: T, result: var Vector) =
  if S.shape[0] != len(vec):
    let msg = fmt"len(vec){len(vec)} != shape[0] {S.shape[0]}."
    raise newException(ValueError, msg)

  result[0..^1] = 0.0
  for i in 0..<S.shape[0]:
    for (j, val) in S.getRow(i):
      result[j] += val * vec[i]


proc vmmul*[T: ColData](vec: Vector, S: T, result: var Vector) =
  if S.shape[0] != len(vec):
    let msg = fmt"len(vec){len(vec)} != shape[0] {S.shape[0]}."
    raise newException(ValueError, msg)

  for j in 0..<S.shape[1]:
    result[j] = 0
    for (i, val) in S.getCol(j):
      result[j] += val * vec[i]


proc mvmul*[T: CSRDataset|CSCDataset|CSRMatrix|CSCMatrix](S: T, vec: Vector): Vector =
  result = zeros([S.shape[0]])
  mvmul(S, vec, result)


proc vmmul*[T: CSRDataset|CSCDataset|CSRMatrix|CSCMatrix](vec: Vector, S: T): Vector =
  result = zeros([S.shape[1]])
  vmmul(vec, S, result)


proc norm*[T: RowData](X: T, p=1, axis=0): Vector =
  if axis == 0:
    result = zeros([X.shape[1]])
    for i in 0..<X.shape[0]:
      for (j, val) in X.getRow(i):
        result[j] += val ^ p
  elif axis == 1:
    result = zeros([X.shape[0]])
    for i in 0..<X.shape[0]:
      for (j, val) in X.getRow(i):
        result[i] += val ^ p
  for i in 0..<len(result):
    result[i] = pow(result[i], 1.0 / float(p))


proc norm*[T: ColData](X: T, p=1, axis=0): Vector =
  if axis == 0:
    result = zeros([X.shape[1]])
    for j in 0..<X.shape[1]:
      for (i, val) in X.getCol(j):
        result[j] += val ^ p
  elif axis == 1:
    result = zeros([X.shape[0]])
    for j in 0..<X.shape[1]:
      for (i, val) in X.getCol(j):
        result[i] += val ^ p
  for i in 0..<len(result):
    result[i] = pow(result[i], 1.0 / float(p))
  