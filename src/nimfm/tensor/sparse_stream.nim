import sequtils, math, strformat, streams, os

const nMagicString* = 9
const magicStringCSR* = ['S', 'T', 'R', 'E', 'A', 'M', 'C', 'S', 'R']
const magicStringCSC* = ['S', 'T', 'R', 'E', 'A', 'M', 'C', 'S', 'C']

type
  SparseElement* = object
    val*: float64
    id*: int


  SparseStreamHeader* = object
    nRows*: int
    nCols*: int
    nnz*: int
    max*: float64
    min*: float64


  BaseSparseStreamMatrix* = ref object of RootObj
    shape: array[2, int]
    nnz: int
    strm: FileStream
    f: string
    max: float64
    min: float64
    coef: float64
    cachesize: int
    indptr: seq[int]
    data*: seq[SparseElement]
    offset: int # first row index in cache
    nCachedVectors: int
 

  StreamCSRMatrix* = ref object of BaseSparseStreamMatrix
    ## Type for row-wise optimizers.


  StreamCSCMatrix* = ref object of BaseSparseStreamMatrix
    ## Type for column-wise optimizers.


## Returns the shape of the matrix.
func shape*(self: BaseSparseStreamMatrix): array[2, int] = self.shape


## Returns the number of non-zero elements in the matrix.
func nnz*(self: BaseSparseStreamMatrix): int = self.nnz


## Returns the maximum value in the matrix.
func max*(self: BaseSparseStreamMatrix): float64 = self.max


## Returns the maximum value in the matrix.
func min*(self: BaseSparseStreamMatrix): float64 = self.min


proc `*=`*(self: BaseSparseStreamMatrix, val: float64) = 
  ## Multiples val to each element (in-place).
  self.coef *= val

proc `/=`*(self: BaseSparseStreamMatrix, val: float64) = 
  ## Divvides each element by val (in-place).
  self.coef *= 1.0 / val


proc newStreamCSRMatrix*(f: string, cacheSize: int =200): StreamCSRMatrix =
  ## Creates new StreamCSRMatrix instance.
  ## filename: a binary file name.
  ## cacheSize: size of cache (MB).
  var header: SparseStreamHeader
  var strm = openFileStream(expandTilde(f), fmRead)
  if strm.isNil:
    raise newException(IOError, fmt"{f} cannot be opened.")
  
  # read/check magic string
  var magic: array[nMagicString, char] # STREAMCSR
  discard strm.readData(addr(magic), nMagicString)
  if magic != magicStringCSR:
    raise newException(IOError, fmt"{f} is not a StreamCSR file.")

  # read header
  discard strm.readData(addr(header), sizeof(header))
  let shape = [header.nRows, header.nCols]
  let nnzAvg = (header.nnz) div header.nRows + 1

  # cachesize = (nnzAvg * sizeof(SparseElement) + sizeof(int)) * nCachedVectors
  #             + sizeof(int)
  let cacheSizeByte = cachesize * (1024^2) # cachesize: MB
  let rowSizeAvg = (nnzAvg*(sizeof(SparseElement)+sizeof(int)))
  let tmp = (cacheSizeByte - sizeof(int)) div rowSizeAvg
  let nCachedVectorsMax = min(min(tmp, high(int) div nnzAvg), header.nRows)
  let data = newSeq[SparseElement](min(nCachedVectorsMax*nnzAvg, header.nnz))
  let indptr = newSeqWith(nCachedVectorsMax+1, 0)
  result = StreamCSRMatrix(shape: shape, nnz: header.nnz, strm:strm, f: f, 
                           max: header.max, min: header.min, coef: 1.0,
                           data: data, cacheSize: cacheSize, indptr: indptr,
                           offset: header.nRows+1, nCachedVectors: 0)


proc newStreamCSCMatrix*(f: string, cacheSize: int=200): StreamCSCMatrix =
  ## Create new StreamCSCMatrix instance.
  ## filename: binary file name.
  ## cacheSize: size of cache (MB).
  var header: SparseStreamHeader
  var strm = openFileStream(expandTilde(f), fmRead)
  if strm.isNil:
    raise newException(IOError, fmt"{f} cannot be opened.")
  # read magic string
  var magic: array[nMagicString, char] # STREAMCSC
  discard strm.readData(addr(magic), nMagicString)
  if magic != magicStringCSC:
    raise newException(IOError, fmt"{f} is not a StreamCSC file.")
  
  # read header
  discard strm.readData(addr(header), sizeof(header))
  let shape = [header.nRows, header.nCols]
  let nnzAvg = (header.nnz) div header.nCols + 1
  
  # cachesize = (nnzAvg * sizeof(SparseElement) + sizeof(int)) * nCachedVectors + sizeof(int)
  let cacheSizeByte =  cachesize * (1024^2) # cachesize: MB
  let colSizeAvg = (nnzAvg*(sizeof(SparseElement)+sizeof(int)))
  let tmp = cacheSizeByte div colSizeAvg
  let nCachedColsMax = min(min(tmp, high(int) div nnzAvg), header.nCols)
  let data = newSeq[SparseElement](min(nCachedColsMax*nnzAvg, header.nnz))
  let indptr = newSeqWith(nCachedColsMax+1, 0)
  
  result = StreamCSCMatrix(shape: shape, nnz: header.nnz, strm: strm, f: f,
                           max: header.max,  min: header.min, coef: 1.0,
                           data: data, cacheSize: cacheSize, indptr: indptr,
                           offset: header.nCols+1, nCachedVectors: 0)


proc readCache(self: BaseSparseStreamMatrix, i: int, transpose=false) =
  let order = if transpose: "col" else: "row"
  if i < self.offset: # read from the first row
    self.strm.setPosition(nMagicString+sizeof(SparseStreamHeader))
    self.offset = 0
    self.nCachedVectors = 0
  var nnz: int
  var jj: int
  while not (i >= self.offset and i <= self.offset+self.nCachedVectors-1):
    inc(self.offset, self.nCachedVectors)
    self.nCachedVectors = 0
    while not self.strm.atEnd():
      if self.nCachedVectors >= len(self.indptr) - 1: # if cache is filled
        break
      self.strm.read(nnz)
      # if cache is filled
      if (nnz + self.indptr[self.nCachedVectors]) > len(self.data):
        self.strm.setPosition(self.strm.getPosition()-sizeof(nnz))
        break
      else: # read a row and cache it
        jj = self.indptr[self.nCachedVectors]
        if nnz > len(self.data):
          let msg = fmt"{self.offset}-th {order} cannot be read." 
          raise newException(ValueError, msg & " Set cacheSize to be larger.")

        discard self.strm.readData(addr(self.data[jj]),
                                   nnz * sizeof(SparseElement))
        self.indptr[self.nCachedVectors+1] = self.indptr[self.nCachedVectors] + nnz
        inc(self.nCachedVectors)
      
    # check error
    if self.nCachedVectors == 0:
      let msg = fmt"{self.offset}-th {order} cannot be read." 
      raise newException(ValueError, msg & " Set cacheSize to be larger.")
    
    if self.strm.atEnd():
      if i < self.offset or i > self.offset+self.nCachedVectors-1:
        self.strm.close()
        let msg = fmt"{i}-th {order} cannot be read. Please check your data format."
        raise newException(IndexError, msg)


iterator getRow*(self: StreamCSRMatrix, i: int): (int, float64) =
  ## Yield the index and the value of non-zero elements in i-th row.
  ## if the i-th row is not in cache, read data and cache them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  if i < self.offset or i > self.offset+self.nCachedVectors-1:
    readCache(self, i)
  
  var jj = self.indptr[i-self.offset]
  let jjMax = self.indptr[i-self.offset+1]
  while (jj < jjMax):
    yield (self.data[jj].id, self.coef*self.data[jj].val)
    inc(jj)


proc getRow*(self: StreamCSRMatrix, i: int): iterator(): (int, float64) =
  ## Yield the index and the value of non-zero elements in i-th row.
  ## if the i-th row is not in cache, read data and cache them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  return iterator(): (int, float64) =
    if i < self.offset or i > self.offset+self.nCachedVectors-1:
      readCache(self, i)
    
    var jj = self.indptr[i-self.offset]
    let jjMax = self.indptr[i-self.offset+1]
    while (jj < jjMax):
      yield (self.data[jj].id, self.coef*self.data[jj].val)
      inc(jj)


iterator getRowIndices*(self: StreamCSRMatrix, i: int): int =
  ## Yield the index of non-zero elements in i-th row.
  ## if the i-th row is not in cache, read data and cache them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  if i < self.offset or i > self.offset+self.nCachedVectors-1:
    readCache(self, i)
  
  var jj = self.indptr[i-self.offset]
  let jjMax = self.indptr[i-self.offset+1]
  while (jj < jjMax):
    yield self.data[jj].id
    inc(jj)


proc getRowIndices*(self: StreamCSRMatrix, i: int): iterator(): int=
  ## Yield the index of non-zero elements in i-th row.
  ## if the i-th row is not in cache, read data and cache them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  return iterator(): int =
    if i < self.offset or i > self.offset+self.nCachedVectors-1:
      readCache(self, i)
    
    var jj = self.indptr[i-self.offset]
    let jjMax = self.indptr[i-self.offset+1]
    while (jj < jjMax):
      yield self.data[jj].id
      inc(jj)


iterator getCol*(self: StreamCSCMatrix, j: int): (int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  ## if the j-th column is not in cache, read data and cache them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)

  if j < self.offset or j > self.offset+self.nCachedVectors-1:
    readCache(self, j, true)
  var ii = self.indptr[j-self.offset]
  let iiMax = self.indptr[j-self.offset+1]
  while (ii < iiMax):
    yield (self.data[ii].id, self.coef*self.data[ii].val)
    inc(ii)


proc getCol*(self: StreamCSCMatrix, j: int): iterator(): (int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  ## if the j-th column is not in cache, read data and cache them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)
 
  return iterator(): (int, float64) =
    if j < self.offset or j > self.offset+self.nCachedVectors-1:
      readCache(self, j, true)
    var ii = self.indptr[j-self.offset]
    let iiMax = self.indptr[j-self.offset+1]
    while (ii < iiMax):
      yield (self.data[ii].id, self.coef*self.data[ii].val)
      inc(ii)


iterator getColIndices*(self: StreamCSCMatrix, j: int): int =
  ## Yields the index of non-zero elements in j-th column.
  ## if the j-th column is not in cache, read data and cache them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)
  
  if j < self.offset or j > self.offset+self.nCachedVectors-1:
    readCache(self, j, true)

  var ii = self.indptr[j-self.offset]
  let iiMax = self.indptr[j-self.offset+1]
  while (ii < iiMax):
    yield self.data[ii].id
    inc(ii)


proc getColIndices*(self: StreamCSCMatrix, j: int): iterator(): int =
  ## Yields the index of non-zero elements in j-th column.
  ## if the j-th column is not in cache, read data and cache them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)

  return iterator(): int =
    if j < self.offset or j > self.offset+self.nCachedVectors-1:
      readCache(self, j, true)
    var ii = self.indptr[j-self.offset]
    let iiMax = self.indptr[j-self.offset+1]
    while (ii < iiMax):
      yield self.data[ii].id
      inc(ii)