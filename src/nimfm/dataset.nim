import sequtils, sugar, parseutils, math, os, strformat


const Integers = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}


type
  BaseDataset* = ref object of RootObj
    nSamples*: int
    nFeatures*: int
    data: seq[float64]
 
  CSRDataset* = ref object of BaseDataset
   indices: seq[int]
   indptr: seq[int]

  CSCDataset* = ref object of BaseDataset
    indices: seq[int]
    indptr: seq[int]

  NormKind* = enum
    l1, l2, linfty


func nnz*(self: BaseDataset): int = len(self.data)


func max*(self: BaseDataset): float64 = max(self.data)


func min*(self: BaseDataset): float64 = min(self.data)


proc `*=`*(self: BaseDataset, val: float64) = 
  for i in 0..<self.nnz:
    self.data[i] *= val


proc `/=`*(self: BaseDataset, val: float64) = self *= 1.0 / val


func sum*(self: BaseDataset): float64 = sum(self.data)


func newCSRDataset*(data: seq[float64], indices, indptr: seq[int], 
                    nSamples, nFeatures: int): CSRDataset =
  result = CSRDataset(data: data, indices: indices, indptr: indptr, 
                      nSamples: nSamples, nFeatures: nFeatures)


func newCSCDataset*(data: seq[float64], indices, indptr: seq[int], 
                    nSamples, nFeatures: int): CSCDataset =
  result = CSCDataset(data: data, indices: indices, indptr: indptr, 
                      nSamples: nSamples, nFeatures: nFeatures)


iterator getRow*(self: CSRDataset, i: int): tuple[j: int, val: float64] =
  var jj = self.indptr[i]
  let jjMax = self.indptr[i+1]
  while (jj < jjMax):
    yield (self.indices[jj], self.data[jj])
    inc(jj)


iterator getCol*(self: CSCDataset, j: int): tuple[i: int, val: float64] =
  var ii = self.indptr[j]
  let iiMax = self.indptr[j+1]
  while (ii < iiMax):
    yield (self.indices[ii], self.data[ii])
    inc(ii)


func `[]`*(self: CSRDataset, i, j: int): float64 =
  result = 0.0
  for (j2, val) in self.getRow(i):
    if j2 == j:
      result = val
      break
    elif j2 > j: break


func `[]`*(self: CSCDataset, i, j: int): float64 =
  result = 0.0
  for (i2, val) in self.getCol(j):
    if i2 == i:
      result = val
      break
    elif i2 > i: break


func checkIndicesRow(X: BaseDataset, indicesRow: openarray[int]) =
  if len(indicesRow) < 1:
    raise newException(ValueError, "Invalid slice: nSamples < 1.")

  if max(indicesRow) >= X.nSamples:
    let msg = fmt"max(indicesRow) {max(indicesRow)} >= {X.nSamples}."
    raise newException(ValueError, msg)
  
  if min(indicesRow) < 0:
    raise newException(ValueError, fmt"min(indicesRow) {min(indicesRow)} < 0.")


func `[]`*(X: CSRDataset, indicesRow: openarray[int]): CSRDataset =
  checkIndicesRow(X, indicesRow)
  let nFeatures = X.nFeatures
  let nSamples = len(indicesRow)
  var indptr = newSeqWith(nSamples+1, 0)
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

  result = newCSRDataset(data, indices, indptr, nSamples, nFeatures)


func `[]`*(X: CSRDataset, slice: Slice[int]): CSRDataset =
  result = X[toSeq(slice)]


func `[]`*(X: CSRDataset, slice: HSlice[int, BackwardsIndex]): CSRDataset =
  result = X[slice.a..(X.nSamples-int(slice.b))]


# Accessing row vectors by openarray is not supported for CSCDataset now
# By Slice is supported, but slow (O(nnz(X)))
func `[]`*(X: CSCDataset, slice: Slice[int]): CSCDataset =
  checkIndicesRow(X, toSeq(slice))
  let nFeatures = X.nFeatures
  let nSamples = slice.b - slice.a+1
  var indptr = newSeqWith(X.nFeatures+1, 0)
  
  for j in 0..<nFeatures:
    for (i, val) in X.getCol(j):
      if i >= slice.a and i <= slice.b:
        inc(indptr[j+1])
  
  cumsum(indptr)
  let nnz = indptr[^1]
  var indices = newSeqWith(nnz, 0)
  var data = newSeqWith(nnz, 0.0)

  for j in 0..<nFeatures:
    var count = 0
    for (i, val) in X.getCol(j):
      if i >= slice.a and i <= slice.b:
        indices[indptr[j] + count] = i - slice.a
        data[indptr[j]+count] = val
        inc(count)
  result = newCSCDataset(data, indices, indptr, nSamples, nFeatures)


func `[]`*(X: CSCDataset, slice: HSlice[int, BackwardsIndex]): CSCDataset =
  result = X[slice.a..(X.nSamples-int(slice.b))]


func shuffle*[T](X: CSRDataset, y: seq[T], indices: openarray[int]):
                 tuple[XShuffled: CSRDataset, yShuffled: seq[T]] =
  var yShuffled = newSeq[T](len(indices))
  for ii, i in indices:
    yShuffled[ii] = y[i]
  
  result = (X[indices], yShuffled)
    

# Normizling function for CSRDataset
proc normalize(data: var seq[float64], indices, indptr: seq[int],
               axis, nRows, nCols: int, f: (float64, float64)->float64,
               g: (float64)->float64) =
  ## f: incremental update function for norm
  ## g: finalize function for norm
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


proc normalize*(X: var CSRDataset, axis=1, norm: NormKind = l2) =
  case norm
  of l2:
    normalize(X.data, X.indices, X.indptr, axis, X.nSamples, X.nFeatures,
              (x, y) => x + y^2, sqrt)
  of l1:
    normalize(X.data, X.indices, X.indptr, axis, X.nSamples, X.nFeatures,
             (x, y) => x + abs(y), x => x)
  of linfty:
    normalize(X.data, X.indices, X.indptr, axis, X.nSamples, X.nFeatures,
              (x, y) => max(x, abs(y)), x => x)


proc normalize*(X: var CSCDataset, axis=1, norm: NormKind = l2) =
  case norm # Leverage the fact that transpose of CSC is CSR
  of l2:
    normalize(X.data, X.indices, X.indptr, 1-axis, X.nFeatures, X.nSamples,
              (x, y) => x + y^2, sqrt)
  of l1:
    normalize(X.data, X.indices, X.indptr, 1-axis, X.nFeatures, X.nSamples,
             (x, y) => x + abs(y), x => x)
  of linfty:
    normalize(X.data, X.indices, X.indptr, 1-axis, X.nFeatures, X.nSamples,
              (x, y) => max(x, abs(y)), x => x)


func toCSR*(input: seq[seq[float64]]): CSRDataset =
  let
    nSamples = len(input)
    nFeatures = len(input[0])
  var
    data: seq[float64]
    indices: seq[int]
    indptr: seq[int] = newSeqWith(nSamples+1, 0)
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      if input[i][j] != 0.0:
        indptr[i+1] += 1

  cumsum(indptr)
  data = newSeqWith(indptr[nSamples], 0.0)
  indices = newSeqWith(indptr[nSamples], 0)
  var count = 0
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      if input[i][j] != 0.0:
        data[count] = input[i][j]
        indices[count] = j
        count += 1
  result = newCSRDataset(
    data=data, indices=indices, indptr=indptr, nSamples=nSamples,
    nFeatures=nFeatures)


func toCSC*(input: seq[seq[float64]]): CSCDataset =
  let
    nSamples = len(input)
    nFeatures = len(input[0])
  var
    data: seq[float64]
    indices: seq[int]
    indptr: seq[int] = newSeqWith(nFeatures+1, 0)

  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      if input[i][j] != 0.0:
        indptr[j+1] += 1

  cumsum(indptr)
  data = newSeqWith(indptr[nFeatures], 0.0)
  indices = newSeqWith(indptr[nFeatures], 0)
  var offsets = newSeqWith(nFeatures, 0)
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      if input[i][j] != 0.0:
        data[indptr[j]+offsets[j]] = input[i][j]
        indices[indptr[j]+offsets[j]] = i
        offsets[j] += 1

  result = newCSCDataset(
    data=data, indices=indices, indptr=indptr, nSamples=nSamples,
    nFeatures=nFeatures)


proc toCSR*(self: CSCDataset): CSRDataset =
  let nSamples = self.nSamples
  let nFeatures = self.nFeatures
  var data = newSeqWith(self.nnz, 0.0)
  var indices = newSeqWith(self.nnz, 0)
  var indptr = newSeqWith(self.nSamples+1, 0)
  
  for j in 0..<nFeatures:
    for (i, val) in self.getCol(j):
      indptr[i+1] += 1
  cumsum(indptr)
  var offsets = newSeqWith(nSamples, 0)
  for j in 0..<nFeatures:
    for (i, val) in self.getCol(j):
      data[indptr[i]+offsets[i]] = val
      indices[indptr[i] + offsets[i]] = j
      offsets[i] += 1
  result = newCSRDataset(data=data, indices=indices, indptr=indptr,
                         nSamples=nSamples, nFeatures=nFeatures)


func toCSC*(self: CSRDataset): CSCDataset =
  let nSamples = self.nSamples
  let nFeatures = self.nFeatures
  var data = newSeqWith(self.nnz, 0.0)
  var indices = newSeqWith(self.nnz, 0)
  var indptr = newSeqWith(self.nFeatures+1, 0)
  
  for i in 0..<nSamples:
    for (j, val) in self.getRow(i):
      indptr[j+1] += 1
  cumsum(indptr)
  var offsets = newSeqWith(nFeatures, 0)
  for i in 0..<nSamples:
    for (j, val) in self.getRow(i):
      data[indptr[j]+offsets[j]] = val
      indices[indptr[j] + offsets[j]] = i
      offsets[j] += 1
  result = newCSCDataset(data=data, indices=indices, indptr=indptr,
                         nSamples=nSamples, nFeatures=nFeatures)


func toSeq*(self: CSRDataset): seq[seq[float64]] = 
  let nSamples = self.nSamples
  let nFeatures = self.nFeatures
  result = newSeqWith(nSamples, newSeqWith(nFeatures, 0.0))
  for i in 0..<nSamples:
    for (j, val) in self.getRow(i):
      result[i][j] = val


func toSeq*(self: CSCDataset): seq[seq[float64]] = 
  let nSamples = self.nSamples
  let nFeatures = self.nFeatures
  result = newSeqWith(nSamples, newSeqWith(nFeatures, 0.0))
  for j in 0..<nSamples:
    for (i, val) in self.getCol(j):
      result[i][j] = val


func vstack*(dataseq: varargs[CSRDataset]): CSRDataset =
  new(result)
  result.nSamples = dataseq[0].nSamples
  result.nFeatures = dataseq[0].nFeatures
  result.data = dataseq[0].data
  result.indices = dataseq[0].indices
  result.indptr = dataseq[0].indptr

  for X in dataseq[1..^1]:
    let nnz = result.nnz
    if X.nFeatures != result.nFeatures:
      raise newException(ValueError, "Different nFeatures.")
    result.data &= X.data
    result.indices &= X.indices
    result.indptr &= X.indptr[1..^1].map(x=>(nnz+x))
    result.nSamples += X.nSamples


func hstack*(dataseq: varargs[CSCDataset]): CSCDataset =
  new(result)
  result.nSamples = dataseq[0].nSamples
  result.nFeatures = dataseq[0].nFeatures
  result.data = dataseq[0].data
  result.indices = dataseq[0].indices
  result.indptr = dataseq[0].indptr
  for X in dataseq[1..^1]:
    let nnz = result.nnz
    if X.nSamples != result.nSamples:
      raise newException(ValueError, "Different nSamples.")
    result.data &= X.data
    result.indices &= X.indices
    result.indptr &= X.indptr[1..^1].map(x=>(nnz+x))
    result.nFeatures += X.nFeatures


proc loadSVMLightFile(f: string, y: var seq[float]):
                      tuple[data: seq[float], indices, indptr: seq[int],
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


proc loadSVMLightFile*(f: string, dataset: var CSRDataset, y: var seq[float],
                       nFeatures: int = -1) =

  var data: seq[float]
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
  var yFloat: seq[float]
  loadSVMLightFile(f, dataset, yFloat, nFeatures)
  y = map(yFloat, proc (x: float): int = int(x))


proc loadSVMLightFile*(f: string, dataset: var CSCDataset, y: var seq[float],
                       nFeatures: int = -1) =

  var data: seq[float]
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
  var val, target: float
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
  var yFloat: seq[float]
  loadSVMLightFile(f, dataset, yFloat, nFeatures)
  y = map(yFloat, proc (x: float): int = int(x))


proc dumpSVMLightFile*(f: string, X: CSRDataset, y: seq[SomeNumber]) =
  var f: File = open(f, fmwrite)
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
  var f: File = open(f, fmwrite)
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


proc loadUserItemRatingFile*(f: string, X: var CSRDataset, y: var seq[float64]) =
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
    start = skipUntil(line, Integers, 0)
    start += parseInt(line, indices[2*i], start)

    start += skipUntil(line, Integers, start)
    start += parseInt(line, indices[2*i+1], start)
    
    start += skipUntil(line, Integers, start)
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


proc loadUserItemRatingFile*(f: string, X: var CSCDataset, y: var seq[float64]) =
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

    start = skipUntil(line, Integers, 0)
    start += parseInt(line, user, start)

    start += skipUntil(line, Integers, start)
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

    start = skipUntil(line, Integers, 0)
    start += parseInt(line, user, start)
    j = user - minUser
    indptr[j+1] += 1

    start += skipUntil(line, Integers, start)
    start += parseInt(line, item, start)
    j = item - minItem + nUsers
    indptr[j+1] += 1

    start += skipUntil(line, Integers, start)
    start += parseFloat(line, y[i], start)
    i += 1
  cumsum(indptr)
  i = 0
  var offsets = newSeqWith(nFeatures, 0)
  for line in f.lines:
    if len(line) < 5:
      continue 

    start = skipUntil(line, Integers, 0)
    start += parseInt(line, user, start)
    j = user - minUser
    indices[indptr[j]+offsets[j]] = i
    offsets[j] += 1

    start += skipUntil(line, Integers, start)
    start += parseInt(line, item, start)
    j = item - minItem + nUsers
    indices[indptr[j]+offsets[j]] = i
    offsets[j] += 1
    i += 1

  X = newCSCDataset(data=data, indices=indices, indptr=indptr,
                    nFeatures=nUsers+nItems, nSamples=nSamples)
