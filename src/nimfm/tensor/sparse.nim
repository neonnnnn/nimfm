import sequtils, sugar, math, strformat


type
  BaseSparseMatrix* = ref object of RootObj
    shape: array[2, int]
    data*: seq[float64]
 
  CSRMatrix* = ref object of BaseSparseMatrix
   ## Type for row-wise optimizers.
   indices*: seq[int]
   indptr*: seq[int]

  CSCMatrix* = ref object of BaseSparseMatrix
    ## Type for column-wise optimizers.
    indices*: seq[int]
    indptr*: seq[int]

  NormKind* = enum
    l1, l2, linfty


func shape*(self: BaseSparseMatrix): array[2, int] =
  ## Returns the shape of the matix.
  result = self.shape


func nnz*(self: BaseSparseMatrix): int =
  ## Returns the number of non-zero elements in the matrix.
  result = len(self.data)


func max*(self: BaseSparseMatrix): float64 =
  ## Returns the maximum value in the matrix.
  result = max(0.0, max(self.data))


func min*(self: BaseSparseMatrix): float64 =
  ## Returns the maximum value in the matrix.
  result = min(0.0, min(self.data))


proc `*=`*(self: BaseSparseMatrix, val: float64) = 
  ## Multiples val to each element (in-place).
  for i in 0..<self.nnz:
    self.data[i] *= val


proc `/=`*(self: BaseSparseMatrix, val: float64) = 
  ## Divides each element by val (in-place).
  self *= 1.0 / val


func sum*(self: BaseSparseMatrix): float64 =
  ## Returns the sum of elements
  result = sum(self.data)


func newCSRMatrix*(data: seq[float64], indices, indptr: seq[int], 
                   shape: array[2, int]): CSRMatrix =
  ## Creates new CSRMatrix instance
  result = CSRMatrix(data: data, indices: indices, indptr: indptr, 
                     shape: shape)


func newCSCMatrix*(data: seq[float64], indices, indptr: seq[int], 
                   shape: array[2, int]): CSCMatrix =
  ## Creates new CSCMatrix instance
  result = CSCMatrix(data: data, indices: indices, indptr: indptr, 
                     shape: shape)


iterator getRow*(self: CSRMatrix, i: int): (int, float64) =
  ## Yields the index and the value of non-zero elements in i-th row.
  var jj = self.indptr[i]
  let jjMax = self.indptr[i+1]
  while (jj < jjMax):
    yield (self.indices[jj], self.data[jj])
    inc(jj)


proc getRow*(self: CSRMatrix, i: int): iterator(): (int, float64) =
  ## Yields the index and the value of non-zero elements in i-th row.
  return iterator(): (int, float64) =
    var jj = self.indptr[i]
    let jjMax = self.indptr[i+1]
    while (jj < jjMax):
      yield (self.indices[jj], self.data[jj])
      inc(jj)


iterator getRowIndices*(self: CSRMatrix, i: int): int =
  ## Yields the index of non-zero elements in i-th row.
  var jj = self.indptr[i]
  let jjMax = self.indptr[i+1]
  while (jj < jjMax):
    yield self.indices[jj]
    inc(jj)


proc getRowIndices*(self: CSRMatrix, i: int): iterator(): int =
  ## Yields the index of non-zero elements in i-th row.
  return iterator(): int =
    var jj = self.indptr[i]
    let jjMax = self.indptr[i+1]
    while (jj < jjMax):
      yield self.indices[jj]
      inc(jj)


iterator getCol*(self: CSCMatrix, j: int): (int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  var ii = self.indptr[j]
  let iiMax = self.indptr[j+1]
  while (ii < iiMax):
    yield (self.indices[ii], self.data[ii])
    inc(ii)


proc getCol*(self: CSCMatrix, j: int): iterator(): (int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  return iterator(): (int, float64) =
    var ii = self.indptr[j]
    let iiMax = self.indptr[j+1]
    while (ii < iiMax):
      yield (self.indices[ii], self.data[ii])
      inc(ii)


iterator getColIndices*(self: CSCMatrix, j: int): int =
  ## Yields the index of non-zero elements in j-th column.
  var ii = self.indptr[j]
  let iiMax = self.indptr[j+1]
  while (ii < iiMax):
    yield self.indices[ii]
    inc(ii)


proc getColIndices*(self: CSCMatrix, j: int): iterator(): int =
  ## Yields the index of non-zero elements in j-th column.
  return iterator(): int = 
    var ii = self.indptr[j]
    let iiMax = self.indptr[j+1]
    while (ii < iiMax):
      yield self.indices[ii]
      inc(ii)


func `[]`*(self: CSRMatrix, i, j: int): float64 =
  ## Returns the value of (i, j) element in matrix
  result = 0.0
  for (j2, val) in self.getRow(i):
    if j2 == j:
      result = val
      break
    elif j2 > j: break


func `[]`*(self: CSCMatrix, i, j: int): float64 =
  ## Returns the value of (i, j) element in matrix
  result = 0.0
  for (i2, val) in self.getCol(j):
    if i2 == i:
      result = val
      break
    elif i2 > i: break


func checkIndicesRow(X: BaseSparseMatrix, indicesRow: openarray[int]) =
  if len(indicesRow) < 1:
    raise newException(ValueError, "Invalid slice: number of rows < 1.")

  if max(indicesRow) >= X.shape[0]:
    let msg = fmt"max(indicesRow) {max(indicesRow)} >= {X.shape[0]}."
    raise newException(ValueError, msg)
  
  if min(indicesRow) < 0:
    raise newException(ValueError, fmt"min(indicesRow) {min(indicesRow)} < 0.")


func `[]`*(X: CSRMatrix, indicesRow: openarray[int]): CSRMatrix =
  ## Returns new CSRMatrix by picking rows from X.
  ## Specifies the indices of row vectors taken out by indicesRow.
  checkIndicesRow(X, indicesRow)
  var shape: array[2, int]
  shape[0] = len(indicesRow)
  shape[1] = X.shape[1]
  var indptr = newSeqWith(len(indicesRow)+1, 0)
  var nnz = 0
  for ii, i in indicesRow:
    nnz += (X.indptr[i+1] - X.indptr[i])
    indptr[ii+1] = nnz
  
  var indices = newSeqWith(nnz, 0)
  var data = newSeqWith(nnz, 0.0)
  var count = 0
  for i in indicesRow:
    for (j, val) in X.getRow(i):
      indices[count] = j
      data[count] = val
      inc(count)

  result = newCSRMatrix(data, indices, indptr, shape)


func `[]`*(X: CSRMatrix, slice: Slice[int]): CSRMatrix =
  ## Takes out row vectors of X and returns new CSRMatrixs by slicing.
  result = X[toSeq(slice)]


func `[]`*(X: CSRMatrix, slice: HSlice[int, BackwardsIndex]): CSRMatrix =
  ## Takes out row vectors of X and returns new CSRMatrixs by slicing.
  result = X[slice.a..(X.shape[0]-int(slice.b))]


# Accessing row vectors by openarray is not supported for CSCMatrix now,
# accessing by only Slice is supported, but slow (O(nnz(X)))
func `[]`*(X: CSCMatrix, slice: Slice[int]): CSCMatrix =
  ## Takes out the row vectors of X and returns new CSCMatrixs by slicing.
  checkIndicesRow(X, toSeq(slice))
  var shape: array[2, int]
  shape[0] = slice.b - slice.a + 1
  shape[1] = X.shape[1]
  var indptr = newSeqWith(X.shape[1]+1, 0)
  
  for j in 0..<shape[1]:
    for (i, val) in X.getCol(j):
      if i >= slice.a and i <= slice.b:
        inc(indptr[j+1])
  
  cumsum(indptr)
  let nnz = indptr[^1]
  var indices = newSeqWith(nnz, 0)
  var data = newSeqWith(nnz, 0.0)

  for j in 0..<shape[1]:
    var count = 0
    for (i, val) in X.getCol(j):
      if i >= slice.a and i <= slice.b:
        indices[indptr[j] + count] = i - slice.a
        data[indptr[j]+count] = val
        inc(count)
  result = newCSCMatrix(data, indices, indptr, shape)


func `[]`*(X: CSCMatrix, slice: HSlice[int, BackwardsIndex]): CSCMatrix =
  ## Takes out row vectors from X and returns new CSCMatrix by slicing.
  result = X[slice.a..(X.shape[0]-int(slice.b))]


# Normizling function for CSRMatrix
proc normalize(data: var seq[float64], indices, indptr: seq[int],
               axis, nRows, nCols: int, f: (float64, float64)->float64,
               g: (float64)->float64) =
  ## f: incremental update function for norm.
  ## g: finalize function for norm.
  if axis == 1:
    for i in 0..<nRows:
      var norm = 0.0
      for val in data[indptr[i]..<indptr[i+1]]:
        norm = f(norm, val)
      norm = g(norm)
      if norm != 0.0:
        for jj in indptr[i]..<indptr[i+1]:
          data[jj] /= norm
  elif axis == 0:
    var norms = newSeqWith(nCols, 0.0)
    for (j, val) in zip(indices, data):
      norms[j] = f(norms[j], val)
    for j in 0..<nCols:
      norms[j] = g(norms[j])
    for jj, j in indices:
      if norms[j] != 0.0:
        data[jj] /= norms[j]


proc normalize*(X: CSRMatrix, axis=1, norm: NormKind = l2) =
  ## Normalizes dataset X by l1, l2, or linfty norm (in-place).
  ## axis=1: normalize per instance.
  ## axis=0: normalize per feature.
  ## dummy_features are ignored.
  case norm
  of l2:
    normalize(X.data, X.indices, X.indptr, axis, X.shape[0],
              X.shape[1], (x, y) => x + y^2, sqrt)
  of l1:
    normalize(X.data, X.indices, X.indptr, axis, X.shape[0], X.shape[1],
             (x, y) => x + abs(y), x => x)
  of linfty:
    normalize(X.data, X.indices, X.indptr, axis, X.shape[0], X.shape[1],
              (x, y) => max(x, abs(y)), x => x)


proc normalize*(X: CSCMatrix, axis=1, norm: NormKind = l2) =
  ## Normalizes dataset X by l1, l2, or linfty norm (in-place).
  ## axis=1: normalize per instance.
  ## axis=0: normalize per feature.
  case norm # Leverage the fact that transpose of CSC is CSR
  of l2:
    normalize(X.data, X.indices, X.indptr, 1-axis, X.shape[1], X.shape[0],
              (x, y) => x + y^2, sqrt)
  of l1:
    normalize(X.data, X.indices, X.indptr, 1-axis, X.shape[1], X.shape[0],
              (x, y) => x + abs(y), x => x)
  of linfty:
    normalize(X.data, X.indices, X.indptr, 1-axis, X.shape[1], X.shape[0],
              (x, y) => max(x, abs(y)), x => x)


func toCSRMatrix*(input: seq[seq[float64]]): CSRMatrix =
  ## Transforms seq[seq[float64]] to CSRMatrix.
  let shape = [len(input), len(input[0])]
  var
    data: seq[float64]
    indices: seq[int]
    indptr: seq[int] = newSeqWith(shape[0]+1, 0)
  for i in 0..<shape[0]:
    if len(input[i]) != shape[1]:
      raise newException(ValueError, "All row vectors must have same length.")
    for j in 0..<shape[1]:
      if input[i][j] != 0.0:
        indptr[i+1] += 1

  cumsum(indptr)
  data = newSeqWith(indptr[^1], 0.0)
  indices = newSeqWith(indptr[^1], 0)
  var count = 0
  for i in 0..<shape[0]:
    if len(input[i]) != shape[1]:
      raise newException(ValueError, "All row vectors must have same length.")
    for j in 0..<shape[1]:
      if input[i][j] != 0.0:
        data[count] = input[i][j]
        indices[count] = j
        count += 1
  result = newCSRMatrix(
    data=data, indices=indices, indptr=indptr, shape=shape)


func toCSCMatrix*(input: seq[seq[float64]]): CSCMatrix =
  ## Transforms seq[seq[float64]] to CSCMatrix.
  let shape = [len(input), len(input[0])]
  var
    data: seq[float64]
    indices: seq[int]
    indptr: seq[int] = newSeqWith(shape[1]+1, 0)

  for i in 0..<shape[0]:
    if len(input[i]) != shape[1]:
      raise newException(ValueError, "All row vectors must have the same length.")
    for j in 0..<shape[1]:
      if input[i][j] != 0.0:
        indptr[j+1] += 1

  cumsum(indptr)
  data = newSeqWith(indptr[^1], 0.0)
  indices = newSeqWith(indptr[^1], 0)
  var offsets = newSeqWith(shape[1], 0)
  for i in 0..<shape[0]:
    for j in 0..<shape[1]:
      if input[i][j] != 0.0:
        data[indptr[j]+offsets[j]] = input[i][j]
        indices[indptr[j]+offsets[j]] = i
        inc(offsets[j])

  result = newCSCMatrix(
    data=data, indices=indices, indptr=indptr, shape=shape)


proc toCSRMatrix*(self: CSCMatrix): CSRMatrix =
  ## Transforms CSCMatrix to CSRMatrix
  var data = newSeqWith(self.nnz, 0.0)
  var indices = newSeqWith(self.nnz, 0)
  var indptr = newSeqWith(self.shape[0]+1, 0)
  
  for j in 0..<self.shape[1]:
    for (i, val) in self.getCol(j):
      indptr[i+1] += 1
  cumsum(indptr)
  var offsets = newSeqWith(self.shape[0], 0)
  for j in 0..<self.shape[1]:
    for (i, val) in self.getCol(j):
      data[indptr[i]+offsets[i]] = val
      indices[indptr[i] + offsets[i]] = j
      offsets[i] += 1
  result = newCSRMatrix(data=data, indices=indices, indptr=indptr,
                        shape=self.shape)


func toCSCMatrix*(self: CSRMatrix): CSCMatrix =
  ## Transforms CSRMatrix to CSCMatrix
  var data = newSeqWith(self.nnz, 0.0)
  var indices = newSeqWith(self.nnz, 0)
  var indptr = newSeqWith(self.shape[1]+1, 0)
  
  for i in 0..<self.shape[0]:
    for (j, val) in self.getRow(i):
      indptr[j+1] += 1
  cumsum(indptr)
  var offsets = newSeqWith(self.shape[1], 0)
  for i in 0..<self.shape[0]:
    for (j, val) in self.getRow(i):
      data[indptr[j]+offsets[j]] = val
      indices[indptr[j] + offsets[j]] = i
      offsets[j] += 1
  result = newCSCMatrix(data=data, indices=indices, indptr=indptr,
                        shape=self.shape)


func toSeq*(self: CSRMatrix): seq[seq[float64]] = 
  ## Transforms CSRMatrix to seq[seq[float64]]
  result = newSeqWith(self.shape[0], newSeqWith(self.shape[1], 0.0))
  for i in 0..<self.shape[0]:
    for (j, val) in self.getRow(i):
      result[i][j] = val


func toSeq*(self: CSCMatrix): seq[seq[float64]] = 
  ## Transforms CSCMatrix to seq[seq[float64]]
  result = newSeqWith(self.shape[0], newSeqWith(self.shape[1], 0.0))
  for j in 0..<self.shape[1]:
    for (i, val) in self.getCol(j):
      result[i][j] = val


func vstack*(dataseq: varargs[CSRMatrix]): CSRMatrix =
  ## Stacks CSRMatrics vertically.
  new(result)
  result.shape[0] = dataseq[0].shape[0]
  result.shape[1] = dataseq[0].shape[1]
  result.data = dataseq[0].data
  result.indices = dataseq[0].indices
  result.indptr = dataseq[0].indptr

  for X in dataseq[1..^1]:
    let nnz = result.nnz
    if X.shape[1] != result.shape[1]:
      raise newException(ValueError, "All matrics must have the same shape[1].")
    result.data &= X.data
    result.indices &= X.indices
    result.indptr &= X.indptr[1..^1].map(x=>(nnz+x))
    result.shape[0] += X.shape[0]


func vstack*(dataseq: varargs[CSCMatrix]): CSCMatrix =
  ## Stacks CSCMatrics vertically.
  new(result)
  result.shape[0] = dataseq[0].shape[0]
  result.shape[1] = dataseq[0].shape[1]
  var nnz = dataseq[0].nnz

  for X in dataseq[1..^1]:
    if X.shape[1] != result.shape[1]:
      raise newException(ValueError, "All matrics must have the same shape[1].")
    nnz += X.nnz
    result.shape[0] += X.shape[0]

  result.data.setLen(nnz)
  result.indices.setLen(nnz)
  result.indptr.setLen(result.shape[0]+1)
  nnz = 0
  for j in 0..<result.shape[1]:
    var offset = 0
    for X in dataseq:
      for (i, val) in X.getCol(j):
        result.data[nnz] = val
        result.indices[nnz] = i + offset
        inc(result.indptr[j+1])
        inc(nnz)

      offset += X.shape[0]
  cumsum(result.indptr)


func hstack*(dataseq: varargs[CSRMatrix]): CSRMatrix =
  ## Stacks CSRMatrics horizontally.
  new(result)
  result.shape[0] = dataseq[0].shape[0]
  result.shape[1] = dataseq[0].shape[1]
  var nnz = dataseq[0].nnz

  for X in dataseq[1..^1]:
    if X.shape[0] != result.shape[0]:
      raise newException(ValueError, "All matrics must have the same shape[0].")
    nnz += X.nnz
    result.shape[1] += X.shape[1]

  result.data.setLen(nnz)
  result.indices.setLen(nnz)
  result.indptr.setLen(result.shape[1]+1)
  nnz = 0
  for i in 0..<result.shape[0]:
    var offset = 0
    for X in dataseq:
      for (j, val) in X.getRow(i):
        result.data[nnz] = val
        result.indices[nnz] = j + offset
        inc(result.indptr[i+1])
        inc(nnz)

      offset += X.shape[1]
  cumsum(result.indptr)


func hstack*(dataseq: varargs[CSCMatrix]): CSCMatrix =
  ## Stacks CSCMatrics horizontally.
  new(result)
  result.shape[0] = dataseq[0].shape[0]
  result.shape[1] = dataseq[0].shape[1]
  result.data = dataseq[0].data
  result.indices = dataseq[0].indices
  result.indptr = dataseq[0].indptr
  for X in dataseq[1..^1]:
    let nnz = result.nnz
    if X.shape[0] != result.shape[0]:
      raise newException(ValueError, "All matrics must have the same shape[0]")
    result.data &= X.data
    result.indices &= X.indices
    result.indptr &= X.indptr[1..^1].map(x=>(nnz+x))
    result.shape[1] += X.shape[1]
