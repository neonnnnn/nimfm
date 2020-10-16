import sequtils, sugar, parseutils, math, os, strformat, random, streams
import tensor/sparse, tensor/tensor, tensor/sparse_stream
export NormKind


const Numbers = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}


type
  BaseDataset*[T] = ref object of RootObj
    nAugments*: int
    dummy: seq[float64]
    data*: T
 
  CSRDataset* = BaseDataset[CSRMatrix]

  CSCDataset* = BaseDataset[CSCMatrix]

  StreamCSRDataset* = BaseDataset[StreamCSRMatrix]

  StreamCSCDataset* = BaseDataset[StreamCSCMatrix]
  
  RowDataset* = CSRDataset|StreamCSRDataset

  ColDataset* = CSCDataset|StreamCSCDataset


func newBaseDataset*[T](data: T): BaseDataset[T] =
  result = BaseDataset[T](data:data, nAugments: 0, dummy: @[])


func nSamples*[T](self: BaseDataset[T]): int = self.data.shape[0]


func nFeatures*[T](self: BaseDataset[T]): int =
  result = self.data.shape[1] + self.nAugments


func shape*[T](self: BaseDataset[T]): array[2, int] =
  ## Returns the shape of the dataset
  result = [self.nSamples, self.nFeatures]


func nnz*[T](self: BaseDataset[T]): int =
  ## Returns the number of non-zero elements in the Dataset
  result = nnz(self.data) + self.nSamples*self.nAugments


func max*[T](self: BaseDataset[T]): float64 =
  ## Returns the maximum value in the Dataset
  result = max(0.0, max(self.data))
  if self.nAugments > 0:
    result = max(result, max(self.dummy[0..<self.nAugments]))


func min*[T](self: BaseDataset[T]): float64 =
  ## Returns the maximum value in the Dataset
  result = min(0.0, min(self.data))
  if self.nAugments > 0:
    result = min(result, min(self.dummy[0..<self.nAugments]))


proc `*=`*[T](self: BaseDataset[T], val: float64) = 
  ## Multiples val to each element (in-place).
  self.data *= val


proc `/=`*[T](self: BaseDataset[T], val: float64) = 
  ## Divides each element by val (in-place).
  self.data *= 1.0 / val


proc addDummyFeature*[T](self: BaseDataset[T], value=1.0, n=1) =
  ## Adds an additional dummy feature.
  if n == 0: discard
  for i in 0..<n:
    if self.nAugments + i < len(self.dummy):
      self.dummy[self.nAugments+i] = value
    else:
      self.dummy.add(value)
  self.nAugments += n


proc removeDummyFeature*[T](self: BaseDataset[T], n=1) =
  ## Adds an additional dummy feature.
  if n == 0: discard
  elif self.nAugments - n >= 0:
    self.nAugments -= n
    self.dummy[self.nAugments..<self.nAugments+n] = 0.0
  else:
    raise newException(ValueError, "No dummy feature.")


func newCSRDataset*(data: seq[float64], indices, indptr: seq[int], 
                    nSamples, nFeatures: int): CSRDataset =
  ## Creates new CSRDataset instance
  let csr = newCSRMatrix(data=data, indices=indices, indptr=indptr,
                         shape=[nSamples, nFeatures])
  result = CSRDataset(data: csr, nAugments: 0, dummy: @[])


func newCSCDataset*(data: seq[float64], indices, indptr: seq[int], 
                    nSamples, nFeatures: int): CSCDataset =
  ## Creates new CSCDataset instance
  let csc = newCSCMatrix(data=data, indices=indices, indptr=indptr,
                         shape=[nSamples, nFeatures])
  result = CSCDataset(data: csc, nAugments: 0, dummy: @[])


func newCSRDataset*(data: CSRMatrix): CSRDataset =
  ## Creates new CSRDataset instance
  result = CSRDataset(data: data, nAugments: 0, dummy: @[])


func newCSCDataset*(data: CSCMatrix): CSCDataset =
  ## Creates new CSCDataset instance
  result = CSCDataset(data: data, nAugments: 0, dummy: @[])


proc newStreamCSRDataset*(f: string, cacheSize=200): StreamCSRDataset =
  ## Creates new StreamCSRDataset instance
  let csr = newStreamCSRMatrix(f=f, cacheSize=cacheSize)
  result = newBaseDataset(data=csr)


proc newStreamCSCDataset*(f: string, cacheSize=200): StreamCSCDataset =
  ## Creates new StreamCSCDataset instance
  let csc = newStreamCSCMatrix(f=f, cacheSize=cacheSize)
  result = newBaseDataset(data=csc)


iterator getRow*(self: RowDataset, i: int): (int, float64) =
  ## Yields the index and the value of non-zero elements in i-th row.
  for (j, val) in self.data.getRow(i):
    yield (j, val)

  if self.nAugments > 0:
    for j in 0..<self.nAugments: # dummy feature
      yield (j+self.data.shape[1], self.dummy[j])


proc getRow*(self: RowDataset, i: int): iterator(): (int, float64) =
  ## Yields the index and the value of non-zero elements in i-th row.
  return iterator(): (int, float64) =
    for (j, val) in self.data.getRow(i):
      yield (j, val)

    if self.nAugments > 0:
      for j in 0..<self.nAugments: # dummy feature
        yield (j+self.data.shape[1], self.dummy[j])


iterator getRowIndices*(self: RowDataset, i: int): int =
  ## Yields the index of non-zero elements in i-th row.
  for j in self.data.getRowIndices(i):
    yield j

  if self.nAugments > 0:
    for j in 0..<self.nAugments: # dummy feature
      yield j+self.data.shape[1]


proc getRowIndices*(self: RowDataset, i: int): iterator(): int =
  ## Yields the index of non-zero elements in i-th row.
  return iterator(): int =
    for j in self.data.getRowIndices(i):
      yield j

    if self.nAugments > 0:
      for j in 0..<self.nAugments: # dummy feature
        yield j+self.data.shape[1]


iterator getCol*(self: ColDataset, j: int): (int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  if j < self.data.shape[1]:
    for (i, val) in self.data.getCol(j):
      yield (i, val)
  elif j < self.data.shape[1]+self.nAugments: # dummy feature
    for i in 0..<self.nSamples:
      yield (i, self.dummy[j-self.data.shape[1]])


proc getCol*(self: ColDataset, j: int): iterator(): (int, float64) = 
  ## Yields the index and the value of non-zero elements in j-th column.
  return iterator(): (int, float64) =
    if j < self.data.shape[1]:
      for (i, val) in self.data.getCol(j):
        yield (i, val)
    elif j < self.data.shape[1]+self.nAugments: # dummy feature
      for i in 0..<self.nSamples:
        yield (i, self.dummy[j-self.data.shape[1]])


iterator getColIndices*(self: ColDataset, j: int): int =
  ## Yields the index of non-zero elements in j-th column.
  if j < self.data.shape[1]:
    for i in self.data.getColIndices(j):
      yield i
  elif j < self.data.shape[1]+self.nAugments: # dummy feature
    for i in 0..<self.nSamples:
      yield i


proc getColIndices*(self: ColDataset, j: int): iterator(): int =
  ## Yields the index of non-zero elements in j-th column.
  return iterator(): int =
    if j < self.data.shape[1]:
      for i in self.data.getColIndices(j):
        yield i
    elif j < self.data.shape[1]+self.nAugments: # dummy feature
      for i in 0..<self.nSamples:
        yield i


func `[]`*(self: CSRDataset, i, j: int): float64 =
  ## Returns the value of (i, j) element in Dataset
  result = 0.0
  for (j2, val) in self.getRow(i):
    if j2 == j:
      result = val
      break
    elif j2 > j: break


func `[]`*(self: CSCDataset, i, j: int): float64 =
  ## Returns the value of (i, j) element in Dataset
  result = 0.0
  for (i2, val) in self.getCol(j):
    if i2 == i:
      result = val
      break
    elif i2 > i: break


func checkIndicesRow[T](X: BaseDataset[T], indicesRow: openarray[int]) =
  if len(indicesRow) < 1:
    raise newException(ValueError, "Invalid slice: nSamples < 1.")

  if max(indicesRow) >= X.nSamples:
    let msg = fmt"max(indicesRow) {max(indicesRow)} >= {X.nSamples}."
    raise newException(ValueError, msg)
  
  if min(indicesRow) < 0:
    raise newException(ValueError, fmt"min(indicesRow) {min(indicesRow)} < 0.")


func `[]`*(X: CSRDataset, indicesRow: openarray[int]): CSRDataset =
  ## Returns new CSRDataset whose row vectors are that of X.
  ## Specifies the indices of row vectors taken out by indicesRow.
  checkIndicesRow(X, indicesRow)
  let data = X.data[indicesRow]
  let dummy = X.dummy
  result = CSRDataset(data: data, nAugments: X.nAugments, dummy: dummy)


func `[]`*(X: CSRDataset, slice: Slice[int]): CSRDataset =
  ## Takes out the row vectors of X and returns new CSRDatasets by slicing.
  result = X[toSeq(slice)]


func `[]`*(X: CSRDataset, slice: HSlice[int, BackwardsIndex]): CSRDataset =
  ## Takes out the row vectors of X and returns new CSRDatasets by slicing.
  result = X[slice.a..(X.nSamples-int(slice.b))]


# Accessing row vectors by openarray is not supported for CSCDataset now
# By Slice is supported, but slow (O(nnz(X)))
func `[]`*(X: CSCDataset, slice: Slice[int]): CSCDataset =
  ## Takes out the row vectors of X and returns new CSCDatasets by slicing.
  checkIndicesRow(X, toSeq(slice))
  let data = X.data[slice]
  let dummy = X.dummy
  result = CSCDataset(data: data, nAugments: X.nAugments, dummy: dummy)


func `[]`*(X: CSCDataset, slice: HSlice[int, BackwardsIndex]): CSCDataset =
  ## Takes out the row vectors of X and returns new CSCDatasets by slicing.
  result = X[slice.a..(X.nSamples-int(slice.b))]


func shuffle*[T](X: CSRDataset, y: seq[T],
                 indices: openarray[int]): (CSRDataset, seq[T]) =
  ## Shuffles the dataset X and target y by using indices.
  if len(y) != X.nSamples:
    raise newException(ValueError, "X.nSamples != len(y)")

  var yShuffled = newSeq[T](len(indices))
  for ii, i in indices:
    yShuffled[ii] = y[i]
  
  result = (X[indices], yShuffled)


func shuffle*[T](X: CSRDataset, y: seq[T]): (CSRDataset, seq[T]) =
  ## Shuffles the dataset X and target y.
  if len(y) != X.nSamples:
    raise newException(ValueError, "X.nSamples != len(y)")
  var indices = toSeq(0..<len(y))
  var j, tmp: int
  for i in 0..<len(y):
    j = rand(len(y)-1-i)
    tmp = indices[i]
    indices[i] = indices[i+j]
    indices[i+j] = tmp
  result = shuffle(X, y, indices)


proc normalize*[T](X: BaseDataset[T], axis=1, norm: NormKind = l2) =
  ## Normalizes dataset X by l1, l2, or linfty norm (in-place).
  ## axis=1: normalize per instance.
  ## axis=0: normalize per feature.
  ## Augmented features are ignored.
  normalize(X.data, axis, norm)

  
func toCSRDataset*(input: seq[seq[float64]]): CSRDataset =
  ## Transforms seq[seq[float64]] to CSRDataset.
  let csr = toCSRMatrix(input)
  result = CSRDataset(data: csr, nAugments: 0, dummy: @[])


func toCSCDataset*(input: seq[seq[float64]]): CSCDataset =
  ## Transforms seq[seq[float64]] to CSCDataset.
  let csc = toCSCMatrix(input)
  result = CSCDataset(data: csc, nAugments: 0, dummy: @[])


proc toCSRDataset*(self: CSCDataset): CSRDataset =
  ## Transforms CSCDataset to CSRDataset
  let csr = toCSRMatrix(self.data)
  result = CSRDataset(data: csr, nAugments: 0, dummy: @[])


func toCSCDataset*(self: CSRDataset): CSCDataset =
  ## Transforms CSRDataset to CSCDataset
  let csc = toCSCMatrix(self.data)
  result = CSCDataset(data: csc, nAugments: 0, dummy: @[])


func toSeq*(self: CSRDataset): seq[seq[float64]] = 
  ## Transforms CSRDataset to seq[seq[float64]]
  let nSamples = self.nSamples
  let nFeatures = self.nFeatures
  result = newSeqWith(nSamples, newSeqWith(nFeatures, 0.0))
  for i in 0..<nSamples:
    for (j, val) in self.getRow(i):
      result[i][j] = val


func vstack*(datasets: varargs[CSRDataset]): CSRDataset =
  ## Stacks CSRDatasets vertically
  var mats = newSeq[CSRMatrix](len(datasets))
  var nSamples = 0
  for i, X in datasets:
    if X.nAugments == 0:
      mats[i] = X.data
    else:
      var dummy = newSeqWith(X.nSamples, newSeqWith(X.nAugments, 0.0))
      for n in 0..<X.nSamples:
        dummy[n] = X.dummy[0..<X.nAugments]
      mats[i] = hstack([X.data, toCSRMatrix(dummy)])
  
  let data = vstack(mats)
  result = newCSRDataset(data)


func vstack*(datasets: varargs[CSCDataset]): CSCDataset =
  ## Stacks CSCDatasets vertically
  var mats = newSeq[CSCMatrix](len(datasets))
  var nSamples = 0
  for i, X in datasets:
    if X.nAugments == 0:
      mats[i] = X.data
    else:
      var dummy = newSeqWith(X.nSamples, newSeqWith(X.nAugments, 0.0))
      for n in 0..<X.nSamples:
        dummy[n] = X.dummy[0..<X.nAugments]
      mats[i] = hstack([X.data, toCSCMatrix(dummy)])
  
  let data = vstack(mats)
  result = newCSCDataset(data)


func hstack*(datasets: varargs[CSRDataset]): CSRDataset =
  ## Stacks CSRDatasets horizontally.
  var mats = newSeq[CSRMatrix](len(datasets))
  var nSamples = 0
  for i, X in datasets:
    if X.nAugments == 0:
      mats[i] = X.data
    else:
      var dummy = newSeqWith(X.nSamples, newSeqWith(X.nAugments, 0.0))
      for n in 0..<X.nSamples:
        dummy[n] = X.dummy[0..<X.nAugments]
      mats[i] = hstack([X.data, toCSRMatrix(dummy)])
  
  let data = hstack(mats)
  result = newCSRDataset(data)


func hstack*(datasets: varargs[CSCDataset]): CSCDataset =
  ## Stacks CSCDatasets horizontally.
  var mats = newSeq[CSCMatrix](len(datasets))
  var nSamples = 0
  for i, X in datasets:
    if X.nAugments == 0:
      mats[i] = X.data
    else:
      var dummy = newSeqWith(X.nSamples, newSeqWith(X.nAugments, 0.0))
      for n in 0..<X.nSamples:
        dummy[n] = X.dummy[0..<X.nAugments]
      mats[i] = hstack([X.data, toCSCMatrix(dummy)])
  
  let data = hstack(mats)
  result = newCSCDataset(data)


func toSeq*(self: CSCDataset): seq[seq[float64]] = 
  ## Transforms CSCDataset to seq[seq[float64]]
  let nSamples = self.nSamples
  let nFeatures = self.nFeatures
  result = newSeqWith(nSamples, newSeqWith(nFeatures, 0.0))
  for j in 0..<nSamples:
    for (i, val) in self.getCol(j):
      result[i][j] = val


proc loadSVMLightFile(f: string, y: var seq[float64]):
                      tuple[data: seq[float64], indices, indptr: seq[int],
                            nFeatures: int, offset: int] =
  var nnz: int = 0
  var nSamples: int = 0
  var i, j, k: int
  var val, target: float64
  var minIndex = 1
  var maxIndex = 0
  for line in expandTilde(f).lines:
    k = 0
    nSamples.inc()
    k.inc(parseFloat(line, target, k))
    k.inc()
    while k < len(line):
      k.inc(parseInt(line, j, k))
      minIndex = min(j, minIndex)
      maxIndex = max(j, maxIndex)
      k.inc()
      k.inc(parseFloat(line, val, k))
      k.inc()
      nnz.inc()

  y.setLen(nSamples)
  if minIndex < 0:
    raise newException(ValueError, "Negative index is included.")
  let offset = if minIndex == 0: 0 else: 1 # basically assume 1-based
  let nFeatures = maxIndex + 1 - offset
  
  var data = newSeq[float64](nnz)
  var indices = newSeq[int](nnz)
  var indptr = newSeq[int](nSamples+1)
  i = 0
  nnz = 0
  indptr[0] = 0
  for line in expandTilde(f).lines:
    indptr[i+1] = indptr[i]
    k = 0
    k.inc(parseFloat(line, target, k))
    y[i] = target
    k.inc()
    while k < len(line):
      k.inc(parseInt(line, j, k))
      indices[nnz] = j-offset
      k.inc()
      k.inc(parseFloat(line, val, k))
      data[nnz] = val
      k.inc()
      nnz.inc()
      indptr[i+1].inc()
    i.inc()
  return (data, indices, indptr, nFeatures, offset)


proc loadSVMLightFile*(f: string, dataset: var CSRDataset, y: var seq[float64],
                       nFeatures: int = -1) =
  ## Loads svmlight/libsvm formt file as CSRDataset and seq[float64].
  var data: seq[float64]
  var indices, indptr: seq[int]
  var nFeaturesPredicted: int
  var offset: int
  (data, indices, indptr, nFeaturesPredicted, offset) = loadSVMLightFile(f, y)
  let nSamples = len(y)
  if (nFeatures > 0) and (nFeaturesPredicted > nFeatures):
    var msg = "nFeatures is " & $nFeatures
    msg &= " but dataset has at least " & $nFeaturesPredicted & " features."
    raise newException(ValueError, msg)

  dataset = newCSRDataset(
    data=data, indices=indices, indptr=indptr, nSamples=nSamples,
    nFeatures=max(nFeaturesPredicted, nFeatures))


proc loadSVMLightFile*(f: string, dataset: var CSRDataset, y: var seq[int],
                       zeroBased: bool = true, nFeatures: int = -1) =
  ## Loads svmlight/libsvm formt file as CSRDataset and seq[int].
  var yFloat: seq[float64]
  loadSVMLightFile(f, dataset, yFloat, nFeatures)
  y = map(yFloat, proc (x: float64): int = int(x))


proc loadSVMLightFile*(f: string, dataset: var CSCDataset, y: var seq[float64],
                       nFeatures: int = -1) =
  ## Loads svmlight/libsvm formt file as CSCDataset and seq[float64].
  var data: seq[float64]
  var indices, indptr: seq[int]
  var nFeaturesPredicted: int
  var offset: int
  (data, indices, indptr, nFeaturesPredicted, offset) = loadSVMLightFile(f, y)
  let nSamples = len(y)
  var indptrCSC: seq[int]
  var offsets: seq[int]
  # create csc indptr
  if (nFeatures > 0) and (nFeaturesPredicted > nFeatures):
    var msg = "nFeatures is " & $nFeatures
    msg &= " but dataset has at least " & $nFeaturesPredicted & " features."
    raise newException(ValueError, msg)
  nFeaturesPredicted = max(nFeaturesPredicted, nFeatures)
  indptrCSC = newSeqWith(nFeaturesPredicted+1, 0)
  offsets = newSeqWith(nFeaturesPredicted, 0)
  for j in indices:
    indptrCSC[j+1].inc()
  indptrCSC.cumsum()
  # convert csr data/indices to csc data/indices in-place
  var k, i, j: int
  var val, target: float64
  i = 0
  for line in expandTilde(f).lines:
    k = 0
    k.inc(parseFloat(line, target, k))
    k.inc()
    while k < len(line):
      k.inc(parseInt(line, j, k))
      j -= offset
      indices[indptrCSC[j]+offsets[j]] = i
      k.inc()
      k.inc(parseFloat(line, val, k))
      data[indptrCSC[j] + offsets[j]] = val
      k.inc()
      offsets[j].inc()
    i.inc()
  dataset = newCSCDataset(
    data=data, indices=indices, indptr=indptrCSC, nSamples=nSamples,
    nFeatures=nFeaturesPredicted)


proc loadSVMLightFile*(f: string, dataset: var CSCDataset, y: var seq[int],
                       nFeatures: int = -1) =
  ## Loads svmlight/libsvm formt file as CSCDataset.
  var yFloat: seq[float64]
  loadSVMLightFile(f, dataset, yFloat, nFeatures)
  y = map(yFloat, proc (x: float64): int = int(x))


proc dumpSVMLightFile*(f: string, X: CSRDataset, y: seq[SomeNumber]) =
  ## Dumps CSRDataset and seq[float64] as svmlight/libsvm format file.
  var f: File = open(f, fmWrite)
  if X.nSamples != len(y):
    raise newException(ValueError, "X.nSamples != len(y).")

  for i in 0..<X.nSamples:
    f.write(y[i])
    for (j, val) in X.getRow(i):
      f.write(fmt" {j+1}:{val}")
    if i+1 != X.nSamples:
      f.write("\n")
  f.close()


proc dumpSVMLightFile*(f: string, X: seq[seq[float64]], y: seq[SomeNumber]) =
  ## Dumps seq[seq[float64]] and seq[SomeNumber] as 
  ## svmlight/libsvm format file.
  var f: File = open(f, fmWrite)
  if len(X) != len(y):
    raise newException(ValueError, "X.nSamples != len(y).")

  for i in 0..<len(X):
    f.write(y[i])
    for j, val in X[i]:
      if val != 0.0:
        f.write(fmt" {j+1}:{val}")
    if i+1 != len(X):
      f.write("\n")
  f.close()


proc loadUserItemRatingFile*(f: string, X: var CSRDataset,
                             y: var seq[float64]) =
  ## Loads user-item-rating matrix file as CSRDataset and seq[float64].
  ## Each row consist of 
  ##    user_id[delimiter]item_id[delimiter]rating[delimiter]comments.
  ## The delmiter must be non-number strings.
  ## For example,
  ## 1 1 5
  ## 1|3|1
  ## First row means "1-th user rates 1-th movie 5 point".
  ## Second row means "1-th user rates 3-th movie 1 point".
  var nSamples = 0
  for line in f.lines:
    if len(line) < 4:
      continue
    nSamples += 1

  var indices = newSeqWith(nSamples*2, 0)
  var indptr = toSeq(0..<nSamples+1)
  apply(indptr, x=>2*x)
  var data = newSeqWith(nSamples*2, 1.0)
  y = newSeqWith(nSamples, 0.0)
  
  var
    minUser = 1
    maxUser = 0
    minItem = 1
    maxItem = 0
  
  var i = 0
  var start = 0
  for line in f.lines:
    if len(line) < 5:
      continue
    start = skipUntil(line, Numbers, 0)
    start += parseInt(line, indices[2*i], start)

    start += skipUntil(line, Numbers, start)
    start += parseInt(line, indices[2*i+1], start)
    
    start += skipUntil(line, Numbers, start)
    start += parseFloat(line, y[i], start)

    minUser = min(minUser, indices[2*i])
    maxUser = max(maxUser, indices[2*i])
    minItem = min(minItem ,indices[2*i+1])
    maxItem = max(maxItem, indices[2*i+1])
    i += 1
  if minUser < 0:
    raise newException(ValueError, "The minimum user id < 0.")
  
  if minItem < 0:
    raise newException(ValueError, "The minimum item id < 0.")
  let nUsers = maxUser - minUser + 1
  let nItems = maxItem - minItem + 1

  for i in 0..<nSamples:
    indices[2*i] -= minUser
    indices[2*i+1] += nUsers - minItem
  X = newCSRDataset(data=data, indices=indices, indptr=indptr,
                    nFeatures=nUsers+nItems, nSamples=nSamples)


proc loadUserItemRatingFile*(f: string, X: var CSCDataset, 
                             y: var seq[float64]) =
  ## Loads user-item-rating matrix file as CSCDataset and seq[float64].
  ## Each row consist of 
  ##    user_id[delimiter]item_id[delimiter]rating[delimiter]comments.
  ## The delmiter must be non-number strings.
  ## For example,
  ## 1 1 5
  ## 1|3|1
  ## First row means "1-th user rates 1-th movie 5 point".
  ## Second row means "1-th user rates 3-th movie 1 point".
  var nSamples = 0
  var user, item, start: int
  var
    minUser = 1
    maxUser = 0
    minItem = 1
    maxItem = 0
  # compute maxUser, minUser, maxItem, minItem, and nFeatures
  for line in f.lines:
    if len(line) < 5:
      continue

    start = skipUntil(line, Numbers, 0)
    start += parseInt(line, user, start)

    start += skipUntil(line, Numbers, start)
    start += parseInt(line, item, start)

    minUser = min(minUser, user)
    maxUser = max(maxUser, user)
    minItem = min(minItem, item)
    maxItem = max(maxItem, item)
    nSamples += 1

  if minUser < 0:
    raise newException(ValueError, "The minimum user id < 0.")
  if minItem < 0:
    raise newException(ValueError, "The minimum item id < 0.")
  
  let nUsers = maxUser - minUser + 1
  let nItems = maxItem - minItem + 1
  let nFeatures = nUsers + nItems

  var indptr = newSeqWith(nFeatures+1, 0)
  var data = newSeqWith(nSamples*2, 1.0)
  var indices = newSeqWith(nSamples*2, 0)
  y = newSeqWith(nSamples, 0.0)
  var i = 0
  var j = 0
  # compute indptr and y
  for line in f.lines:
    if len(line) < 5:
      continue

    start = skipUntil(line, Numbers, 0)
    start += parseInt(line, user, start)
    j = user - minUser
    indptr[j+1] += 1

    start += skipUntil(line, Numbers, start)
    start += parseInt(line, item, start)
    j = item - minItem + nUsers
    indptr[j+1] += 1

    start += skipUntil(line, Numbers, start)
    start += parseFloat(line, y[i], start)
    i += 1
  cumsum(indptr)
  i = 0
  var offsets = newSeqWith(nFeatures, 0)
  for line in f.lines:
    if len(line) < 5:
      continue 

    start = skipUntil(line, Numbers, 0)
    start += parseInt(line, user, start)
    j = user - minUser
    indices[indptr[j]+offsets[j]] = i
    offsets[j] += 1

    start += skipUntil(line, Numbers, start)
    start += parseInt(line, item, start)
    j = item - minItem + nUsers
    indices[indptr[j]+offsets[j]] = i
    offsets[j] += 1
    i += 1

  X = newCSCDataset(data=data, indices=indices, indptr=indptr,
                    nFeatures=nUsers+nItems, nSamples=nSamples)


proc loadStreamLabel*(fIn: string, nSamples: int): seq[float64] =
  result = newSeqWith(nSamples, 0.0)
  var strm = openFileStream(expandTIlde(fIn), fmRead)
  var val: float64
  for i in 0..<nSamples:
    strm.read(val)
    result.add(val)
  if not strm.atEnd:
    echo(fmt"nSamples={nSamples} values have been read but some values are left.")
  strm.close()


proc loadStreamLabel*(fIn: string): seq[float64] =
  result = newSeqWith(0, 0.0)
  var strm = openFileStream(expandTIlde(fIn), fmRead)
  var val: float64
  while not strm.atEnd:
    strm.read(val)
    result.add(val)
  strm.close()


proc convertSVMLightFile*(fIn: string, fOutX: string, fOutY: string) =
  ### Converts an SVMLightFile to a binary file (0-based index).
  var
    val: float64
    minVal = high(float64)
    maxVal = low(float64)
    pos, j: int
    minIndex = 1
    maxIndex = 0
    nSamples = 0
    nnz = 0
    strmIn = openFileStream(expandTilde(fIn), fmRead)
    strmOutX = openFileStream(expandTilde(fOutX), fmWrite)
    strmOutY = openFileStream(expandTilde(fOutY), fmWrite)
    target: float64

  if strmIn.isNil:
    raise newException(ValueError, fmt"{fIn} cannot be read.")
  if strmOutX.isNil:
    raise newException(ValueError, fmt"{fOutX} cannot be read.")
  if strmOutY.isNil:
    raise newException(ValueError, fmt"{fOutY} cannot be read.")

  # determine header information
  for line in strmIn.lines:
    pos = 0
    nSamples.inc()
    pos.inc(parseFloat(line, target, pos))
    pos.inc()
    while pos < len(line):
      pos.inc(parseInt(line, j, pos))
      minIndex = min(j, minIndex)
      maxIndex = max(j, maxIndex)
      pos.inc()
      pos.inc(parseFloat(line, val, pos))
      minVal = min(minVal, val)
      maxVal = max(maxVal, val)
      pos.inc()
      nnz.inc()
  if minIndex < 0:
    raise newException(ValueError, "Negative index is included.")
  let nFeatures = maxIndex - minIndex + 1
  var header = SparseStreamHeader(nRows: nSamples, nCols: nFeatures, nnz: nnz,
                                  max: maxVal, min: minVal)
  var element = SparseElement(val: 0.0, id: 0)
  # read again and write
  strmOutX.write(magicStringCSR)
  strmOutX.writeData(addr(header), sizeof(header))
  strmIn.setPosition(0)
  for line in strmIn.lines:
    # determine nnz of line
    pos = 0
    var nnzRow = 0
    pos.inc(parseFloat(line, target, pos))
    pos.inc()
    # count nnz of each row
    while pos < len(line):
      pos.inc(parseInt(line, j, pos))
      pos.inc()
      pos.inc(parseFloat(line, val, pos))
      pos.inc()
      nnz.inc()
      inc(nnzRow)
    # parse string and write
    strmOutX.write(nnzRow)
    pos = 0
    pos.inc(parseFloat(line, target, pos))
    pos.inc()
    strmOutY.write(target)
    while pos < len(line):
      pos.inc(parseInt(line, element.id, pos))
      pos.inc()
      pos.inc(parseFloat(line, element.val, pos))
      pos.inc()
      nnz.inc()
      element.id -= minIndex
      strmOutX.write(element)

  strmIn.close()
  strmOutX.close()
  strmOutY.close()


proc transposeFile*(fIn: string, fOut:string, cacheSize=200) =
  ## Transposes a StreamCSC/CSR and write it.
  var
    strmIn = openFileStream(expandTilde(fIn), fmRead)
    strmOut = openFileStream(expandTilde(fOut), fmWrite)

  if strmIn.isNil:
    raise newException(ValueError, fmt"{fIn} cannot be read.")
  if strmOut.isNil:
    raise newException(ValueError, fmt"{fOut} cannot be read.")
  var header: SparseStreamHeader
  var magic: array[nMagicString, char]
  discard strmIn.readData(addr(magic), nMagicString)

  if magic == magicStringCSR:
    strmOut.write(magicStringCSC)
  elif magic == magicStringCSC:
    strmOut.write(magicStringCSR)
  else:
    let msg = fmt"{fIn} is not neither StreamCSC and StreamCSR file."
    raise newException(IOError, msg)
  
  discard strmIn.readData(addr(header), sizeof(header))
  
  strmOut.write(header)

  # read and write
  if magic == magicStringCSC:
    swap(header.nCols, header.nRows)
  var nnz: int
  var nnzCols = newSeqWith(header.nCols, 0)
  var nnzColsRest = newSeqWith(header.nCols, 0)
  var offsets = newSeqWith(header.nCols, 0)
  var element: SparseElement
  let tmp = (cacheSize * 1024^2 - 3*sizeof(int)*header.nCols)
  var nCachedElementsMax = tmp div sizeof(SparseElement)
  var elements = newSeq[SparseElement](nCachedElementsMax)
  # determine the nnz of each col
  for i in 0..<header.nRows:
    strmIn.read(nnz)
    for j in 0..<nnz:
      strmIn.read(element)
      inc(nnzCols[element.id])
  
  for j in 0..<header.nCols:
    nnzColsRest[j] = nnzCols[j]
  # write file
  var j = 0
  var lastRead = -1

  strmOut.write(nnzCols[0])
  while j < header.nCols:
    var nCachedCols = 0
    strmIn.setPosition(nMagicString+sizeof(SparseStreamHeader))
    # determine the number of cols read
    # in this loop, j..<(j+nCachedCols) columns are dumped completely
    var nCachedRest = nCachedElementsMax
    while j + nCachedCols < header.nCols:
      offsets[j+nCachedCols] = nCachedElementsMax - nCachedRest
      nCachedRest -= nnzColsRest[j+nCachedCols]
      if nCachedRest >= 0:
        nnzColsRest[j+nCachedCols] = 0
        inc(nCachedCols)
      else:
        nnzColsRest[j+nCachedCols] = -nCachedRest
        break
    
    # read rows and caches
    var nCached = 0
    var lastReadNext = 0
    for i in 0..<header.nRows:
      if nCached == nCachedElementsMax:
        break
      var nnzRow = 0
      strmIn.read(nnzRow)
      for _ in 0..<nnzRow:
        strmIn.read(element)
        if element.id >= j and element.id <= (j+nCachedCols):
          if element.id == j and i <= lastRead: # already read
            continue
          if element.id == (j+nCachedCols):
            if offsets[element.id] >= nCachedElementsMax: # filled
              continue
            else:
              lastReadNext = i
          elements[offsets[element.id]].id = i
          elements[offsets[element.id]].val = element.val
          inc(offsets[element.id])
          inc(nCached)
    # write!
    for ii in 0..<nCached:
      strmOut.write(elements[ii])
      dec(nnzCols[j])
      while j < header.nCols and nnzCols[j] == 0:
        inc(j)
        if j < header.nCols:
          strmOut.write(nnzCols[j])
    lastRead = lastReadNext
  strmIn.close()
  strmOut.close()