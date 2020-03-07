import sequtils, sugar, parseutils, math, os

type
  BaseDataset* = ref object of RootObj
    nSamples*: int
    nFeatures*: int
    data: seq[float64]
 
  CSRDataset* = ref object of BaseDataset
   indices: seq[int]
   indptr: seq[int]
   jj, jjMax: int

  CSCDataset* = ref object of BaseDataset
    indices: seq[int]
    indptr: seq[int]
    ii, iiMax: int


proc nnz*(self: BaseDataset): int = len(self.data)


proc max*(self: BaseDataset): float64 = max(self.data)


proc min*(self: BaseDataset): float64 = min(self.data)


proc `*=`*(self: BaseDataset, val: float64) = 
  for i in 0..<self.nnz:
    self.data[i] *= val


proc `/=`*(self: BaseDataset, val: float64) = self *= 1.0 / val


proc sum*(self: BaseDataset): float64 = sum(self.data)


iterator getRow*(dataset: CSRDataset, i: int): tuple[j: int, val: float64] =
  dataset.jj = dataset.indptr[i]
  dataset.jjMax = dataset.indptr[i+1]
  while (dataset.jj < dataset.jjMax):
    yield (dataset.indices[dataset.jj], dataset.data[dataset.jj])
    inc(dataset.jj)


iterator getCol*(dataset: CSCDataset, j: int): tuple[i: int, val: float64] =
  dataset.ii = dataset.indptr[j]
  dataset.iiMax = dataset.indptr[j+1]
  while (dataset.ii < dataset.iiMax):
    yield (dataset.indices[dataset.ii], dataset.data[dataset.ii])
    inc(dataset.ii)


proc toCSR*(input: seq[seq[float64]]): CSRDataset =
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
  var nnz = 0
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      if input[i][j] != 0.0:
        data[nnz] = input[i][j]
        indices[nnz] = j
        nnz += 1
  result = CSRDataset(
    data: data, indices: indices, indptr: indptr, nSamples: nSamples,
    nFeatures: nFeatures, jj: 0, jjMax: 0)


proc toCSC*(input: seq[seq[float64]]): CSCDataset =
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

  result = CSCDataset(
    data: data, indices: indices, indptr: indptr, nSamples: nSamples,
    nFeatures: nFeatures, ii: 0, iiMax: 0)


proc vstack*(dataseq: varargs[CSRDataset]): CSRDataset =
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


proc hstack*(dataseq: varargs[CSCDataset]): CSCDataset =
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

  dataset = CSRDataset(
    data: data, indices: indices, indptr: indptr, nSamples: nSamples,
    nFeatures: max(nFeaturesPredicted, nFeatures), jj: 0, jjMax: 0)


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
  dataset = CSCDataset(
    data: data, indices: indices, indptr: indptrCSC, nSamples: nSamples,
    nFeatures: nFeaturesPredicted, ii: 0, iiMax: 0)


proc loadSVMLightFile*(f: string, dataset: var CSCDataset, y: var seq[int],
                       nFeatures: int = -1) =
  var yFloat: seq[float]
  loadSVMLightFile(f, dataset, yFloat, nFeatures)
  y = map(yFloat, proc (x: float): int = int(x))
