import sequtils, math, strformat, streams, os

const nMagicString* = 9
const magicStringCSR* = ['S', 'T', 'R', 'E', 'A', 'M', 'C', 'S', 'R']
const magicStringCSC* = ['S', 'T', 'R', 'E', 'A', 'M', 'C', 'S', 'C']
const nMagicStringField* = 14
const magicStringCSRField* = ['S', 'T', 'R', 'E', 'A', 'M', 'C', 'S', 'R', 'F', 'I', 'E', 'L', 'D']
const magicStringCSCField* = ['S', 'T', 'R', 'E', 'A', 'M', 'C', 'S', 'C', 'F', 'I', 'E', 'L', 'D']

type
  SparseElement* = object
    val*: float64
    id*: int

  SparseFieldElement* = object
    field*: int
    val*: float64
    id*: int

  SparseStreamHeader* = object
    nRows*: int
    nCols*: int
    nnz*: int
    max*: float64
    min*: float64

  SparseStreamFieldHeader* = object
    nRows*: int
    nCols*: int
    nnz*: int
    nFields*: int
    max*: float64
    min*: float64

  BaseSparseStreamMatrix*[T, U] = ref object of RootObj
    shape: array[2, int]
    header: U
    strm: FileStream
    f: string
    coef: float64
    cachesize: int
    indptr: seq[int]
    data*: seq[T]
    offset: int # first row index in cache
    nCached: int
    nMagicString: int
 
  StreamCSRMatrix* = ref object of BaseSparseStreamMatrix[SparseElement, SparseStreamHeader]
    ## Type for row-wise optimizers.

  StreamCSCMatrix* = ref object of BaseSparseStreamMatrix[SparseElement, SparseStreamHeader]
    ## Type for column-wise optimizers.

  StreamCSRFieldMatrix* = ref object of BaseSparseStreamMatrix[SparseFieldElement, SparseStreamFieldHeader]
    ## Type for row-wise optimizers.

  StreamCSCFieldMatrix* = ref object of BaseSparseStreamMatrix[SparseFieldElement, SparseStreamFieldHeader]
    ## Type for column-wise optimizers.

  StreamRowMatrix* = StreamCSRMatrix|StreamCSRFieldMatrix

  StreamColMatrix* = StreamCSCMatrix|StreamCSCFieldMatrix


## Returns the shape of the matrix.
func shape*[T, U](self: BaseSparseStreamMatrix[T, U]): array[2, int] = self.shape


## Returns the number of non-zero elements in the matrix.
func nnz*[T, U](self: BaseSparseStreamMatrix[T, U]): int = self.header.nnz


## Returns the maximum value in the matrix.
func max*[T, U](self: BaseSparseStreamMatrix[T, U]): float64 = self.header.max


## Returns the maximum value in the matrix.
func min*[T, U](self: BaseSparseStreamMatrix[T, U]): float64 = self.header.min


func nCached*[T, U](self: BaseSparseStreamMatrix[T, U]): int = self.nCached


func nFields*(self: StreamCSRFieldMatrix): int = self.header.nFields


proc `*=`*(self: BaseSparseStreamMatrix, val: float64) = 
  ## Multiples val to each element (in-place).
  self.coef *= val


proc `/=`*(self: BaseSparseStreamMatrix, val: float64) = 
  ## Divvides each element by val (in-place).
  self.coef *= 1.0 / val


proc newStreamCSRMatrix*(f: string, cacheSize: int=200): StreamCSRMatrix =
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

  # cachesize = (nnzAvg * sizeof(SparseElement) + sizeof(int)) * nCached
  #             + sizeof(int)
  let cacheSizeByte = cachesize * (1024^2) # cachesize: MB
  let rowSizeAvg = (nnzAvg*(sizeof(SparseElement)+sizeof(int)))
  let tmp = (cacheSizeByte - sizeof(int)) div rowSizeAvg
  let nCachedMax = min(min(tmp, high(int) div nnzAvg), header.nRows)
  let data = newSeq[SparseElement](min(nCachedMax*nnzAvg, header.nnz))
  let indptr = newSeqWith(nCachedMax+1, 0)
  result = StreamCSRMatrix(shape: shape, header: header, strm:strm, f: f, 
                           coef: 1.0, data: data, cacheSize: cacheSize,
                           indptr: indptr, offset: header.nRows+1,
                           nCached: 0, nMagicString: nMagicString)


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
  
  # cachesize = (nnzAvg * sizeof(SparseElement) + sizeof(int)) * nCached + sizeof(int)
  let cacheSizeByte =  cachesize * (1024^2) # cachesize: MB
  let colSizeAvg = (nnzAvg*(sizeof(SparseElement)+sizeof(int)))
  let tmp = cacheSizeByte div colSizeAvg
  let nCachedColsMax = min(min(tmp, high(int) div nnzAvg), header.nCols)
  let data = newSeq[SparseElement](min(nCachedColsMax*nnzAvg, header.nnz))
  let indptr = newSeqWith(nCachedColsMax+1, 0)
  
  result = StreamCSCMatrix(shape: shape, header: header, strm: strm, f: f,
                           coef: 1.0, data: data, cacheSize: cacheSize,
                           indptr: indptr, offset: header.nCols+1,
                           nCached: 0, nMagicString: nMagicString)


proc newStreamCSRFieldMatrix*(f: string, cacheSize: int =200): StreamCSRFieldMatrix =
  ## Creates new StreamCSRMatrix instance.
  ## filename: a binary file name.
  ## cacheSize: size of cache (MB).
  var header: SparseStreamFieldHeader
  var strm = openFileStream(expandTilde(f), fmRead)
  if strm.isNil:
    raise newException(IOError, fmt"{f} cannot be opened.")
  
  # read/check magic string
  var magic: array[nMagicStringField, char] # STREAMCSRField
  discard strm.readData(addr(magic), nMagicStringField)
  if magic != magicStringCSRField:
    raise newException(IOError, fmt"{f} is not a StreamCSRField file.")

  # read header
  discard strm.readData(addr(header), sizeof(header))
  let shape = [header.nRows, header.nCols]
  let nnzAvg = (header.nnz) div header.nRows + 1

  # cachesize = (nnzAvg * sizeof(SparseElement) + sizeof(int)) * nCached
  #             + sizeof(int)
  let cacheSizeByte = cachesize * (1024^2) # cachesize: MB
  let rowSizeAvg = (nnzAvg*(sizeof(SparseFieldElement)+sizeof(int)))
  let tmp = (cacheSizeByte - sizeof(int)) div rowSizeAvg
  let nCachedMax = min(min(tmp, high(int) div nnzAvg), header.nRows)
  let data = newSeq[SparseFieldElement](min(nCachedMax*nnzAvg, header.nnz))
  let indptr = newSeqWith(nCachedMax+1, 0)
  result = StreamCSRFieldMatrix(shape: shape, header: header, strm:strm, f: f, 
                                coef: 1.0, data: data, cacheSize: cacheSize,
                                indptr: indptr, offset: header.nRows+1,
                                nCached: 0, nMagicString: nMagicStringField)


proc newStreamCSCFieldMatrix*(f: string, cacheSize: int=200): StreamCSCFieldMatrix =
  ## Creates new StreamCSCFieldMatrix instance.
  ## filename: a binary file name.
  ## cacheSize: size of cache (MB).
  var header: SparseStreamFieldHeader
  var strm = openFileStream(expandTilde(f), fmRead)
  if strm.isNil:
    raise newException(IOError, fmt"{f} cannot be opened.")
  # read magic string
  var magic: array[nMagicStringField, char] # STREAMCSC
  discard strm.readData(addr(magic), nMagicStringField)
  if magic != magicStringCSCField:
    raise newException(IOError, fmt"{f} is not a StreamCSCField file.")
  
  # read header
  discard strm.readData(addr(header), sizeof(header))
  let shape = [header.nRows, header.nCols]
  let nnzAvg = (header.nnz) div header.nCols + 1
  
  # cachesize = (nnzAvg * sizeof(SparseElement)
  #             + sizeof(int)) * nCached + sizeof(int)
  let cacheSizeByte =  cachesize * (1024^2) # cachesize: MB
  let colSizeAvg = (nnzAvg*(sizeof(SparseFieldElement)+sizeof(int)))
  let tmp = cacheSizeByte div colSizeAvg
  let nCachedColsMax = min(min(tmp, high(int) div nnzAvg), header.nCols)
  let data = newSeq[SparseFieldElement](min(nCachedColsMax*nnzAvg, header.nnz))
  let indptr = newSeqWith(nCachedColsMax+1, 0)

  result = StreamCSCFieldMatrix(shape: shape, header: header, strm:strm, f: f, 
                                coef: 1.0, data: data, cacheSize: cacheSize,
                                indptr: indptr, offset: header.nCols+1,
                                nCached: 0, nMagicString: nMagicStringField)


proc readCache*[T, U](self: BaseSparseStreamMatrix[T, U], i: int, transpose=false) =
  let order = if transpose: "col" else: "row"
  if i < self.offset: # read from the first row
    self.strm.setPosition(self.nMagicString+sizeof(U))
    self.offset = 0
    self.nCached = 0
  var nnz: int
  var jj: int
  while not (i >= self.offset and i <= self.offset+self.nCached-1):
    inc(self.offset, self.nCached)
    self.nCached = 0
    while not self.strm.atEnd():
      if self.nCached >= len(self.indptr) - 1: # if cache is filled
        break
      self.strm.read(nnz)
      # if cache is filled
      if (nnz + self.indptr[self.nCached]) > len(self.data):
        self.strm.setPosition(self.strm.getPosition()-sizeof(nnz))
        break
      else: # read a row and cache it
        jj = self.indptr[self.nCached]
        if nnz > len(self.data):
          let msg = fmt"{self.offset}-th {order} cannot be read." 
          raise newException(ValueError, msg & " Set cacheSize to be larger.")

        discard self.strm.readData(addr(self.data[jj]), nnz * sizeof(T))
        self.indptr[self.nCached+1] = self.indptr[self.nCached] + nnz
        inc(self.nCached)
      
    # check error
    if self.nCached == 0:
      let msg = fmt"{self.offset}-th {order} cannot be read." 
      raise newException(ValueError, msg & " Set cacheSize to be larger.")
    
    if self.strm.atEnd():
      if i < self.offset or i > self.offset+self.nCached-1:
        self.strm.close()
        let msg = fmt"{i}-th {order} cannot be read. Please check your data format."
        raise newException(IndexError, msg)


iterator getRow*(self: StreamRowMatrix, i: int): (int, float64) =
  ## Yield the index and the value of non-zero elements in i-th row.
  ## If the i-th row is not in cache, this reads data and caches them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  if i < self.offset or i > self.offset+self.nCached-1:
    readCache(self, i)
  
  var jj = self.indptr[i-self.offset]
  let jjMax = self.indptr[i-self.offset+1]
  while (jj < jjMax):
    yield (self.data[jj].id, self.coef*self.data[jj].val)
    inc(jj)


proc getRow*(self: StreamRowMatrix, i: int): iterator(): (int, float64) =
  ## Yield the index and the value of non-zero elements in i-th row.
  ## If the i-th row is not in cache, this reads data and caches them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  return iterator(): (int, float64) =
    if i < self.offset or i > self.offset+self.nCached-1:
      readCache(self, i)
    
    var jj = self.indptr[i-self.offset]
    let jjMax = self.indptr[i-self.offset+1]
    while (jj < jjMax):
      yield (self.data[jj].id, self.coef*self.data[jj].val)
      inc(jj)


iterator getRowIndices*(self: StreamRowMatrix, i: int): int =
  ## Yield the index of non-zero elements in i-th row.
  ## If the i-th row is not in cache, this reads data and caches them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  if i < self.offset or i > self.offset+self.nCached-1:
    readCache(self, i)
  
  var jj = self.indptr[i-self.offset]
  let jjMax = self.indptr[i-self.offset+1]
  while (jj < jjMax):
    yield self.data[jj].id
    inc(jj)


proc getRowIndices*(self: StreamRowMatrix, i: int): iterator(): int=
  ## Yield the index of non-zero elements in i-th row.
  ## If the i-th row is not in cache, this reads data and caches them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  return iterator(): int =
    if i < self.offset or i > self.offset+self.nCached-1:
      readCache(self, i)
    
    var jj = self.indptr[i-self.offset]
    let jjMax = self.indptr[i-self.offset+1]
    while (jj < jjMax):
      yield self.data[jj].id
      inc(jj)


iterator getRowWithField*(self: StreamCSRFieldMatrix, i: int): (int, int, float64) =
  ## Yield the index and the value of non-zero elements in i-th row.
  ## If the i-th row is not in cache, this reads data and caches them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  if i < self.offset or i > self.offset+self.nCached-1:
    readCache(self, i)
  
  var jj = self.indptr[i-self.offset]
  let jjMax = self.indptr[i-self.offset+1]
  while (jj < jjMax):
    yield (self.data[jj].field, self.data[jj].id, self.coef*self.data[jj].val)
    inc(jj)


proc getRowWithField*(self: StreamRowMatrix, i: int): iterator(): (int, int, float64) =
  ## Yield the index and the value of non-zero elements in i-th row.
  ## If the i-th row is not in cache, this reads data and caches them.
  if i < 0:
    raise newException(IndexError, fmt"index out of bounds: {i} < 0.")
  if i >= self.shape[0]:
    let msg = fmt"index out of bounds: {i} >= {self.shape[0]}."
    raise newException(IndexError, msg)

  return iterator(): (int, int, float64) =
    if i < self.offset or i > self.offset+self.nCached-1:
      readCache(self, i)
    
    var jj = self.indptr[i-self.offset]
    let jjMax = self.indptr[i-self.offset+1]
    while (jj < jjMax):
      yield (self.data[jj].field, self.data[jj].id, self.coef*self.data[jj].val)
      inc(jj)


iterator getCol*(self: StreamColMatrix, j: int): (int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  ## If the j-th column is not in cache, this reads data and caches them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)

  if j < self.offset or j > self.offset+self.nCached-1:
    readCache(self, j, true)
  var ii = self.indptr[j-self.offset]
  let iiMax = self.indptr[j-self.offset+1]
  while (ii < iiMax):
    yield (self.data[ii].id, self.coef*self.data[ii].val)
    inc(ii)


proc getCol*(self: StreamColMatrix, j: int): iterator(): (int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  ## If the j-th column is not in cache, this reads data and caches them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)
 
  return iterator(): (int, float64) =
    if j < self.offset or j > self.offset+self.nCached-1:
      readCache(self, j, true)
    var ii = self.indptr[j-self.offset]
    let iiMax = self.indptr[j-self.offset+1]
    while (ii < iiMax):
      yield (self.data[ii].id, self.coef*self.data[ii].val)
      inc(ii)


iterator getColIndices*(self: StreamColMatrix, j: int): int =
  ## Yields the index of non-zero elements in j-th column.
  ## If the j-th column is not in cache, this reads data and caches them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)
  
  if j < self.offset or j > self.offset+self.nCached-1:
    readCache(self, j, true)

  var ii = self.indptr[j-self.offset]
  let iiMax = self.indptr[j-self.offset+1]
  while (ii < iiMax):
    yield self.data[ii].id
    inc(ii)


proc getColIndices*(self: StreamColMatrix, j: int): iterator(): int =
  ## Yields the index of non-zero elements in j-th column.
  ## If the j-th column is not in cache, this reads data and caches them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)

  return iterator(): int =
    if j < self.offset or j > self.offset+self.nCached-1:
      readCache(self, j, true)
    var ii = self.indptr[j-self.offset]
    let iiMax = self.indptr[j-self.offset+1]
    while (ii < iiMax):
      yield self.data[ii].id
      inc(ii)


iterator getColWithField*(self: StreamCSCFieldMatrix, j: int): (int, int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  ## If the j-th column is not in cache, this reads data and caches them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)

  if j < self.offset or j > self.offset+self.nCached-1:
    readCache(self, j, true)
  var ii = self.indptr[j-self.offset]
  let iiMax = self.indptr[j-self.offset+1]
  while (ii < iiMax):
    yield (self.data[ii].field, self.data[ii].id, self.coef*self.data[ii].val)
    inc(ii)


proc getColWithField*(self: StreamCSCFieldMatrix, j: int): iterator(): (int, int, float64) =
  ## Yields the index and the value of non-zero elements in j-th column.
  ## If the j-th column is not in cache, this reads data and caches them.
  if j < 0:
    raise newException(IndexError, fmt"index out of bounds: {j} < 0.")
  if j >= self.shape[1]:
    let msg = fmt"index out of bounds: {j} >= {self.shape[1]}."
    raise newException(IndexError, msg)
 
  return iterator(): (int, int, float64) =
    if j < self.offset or j > self.offset+self.nCached-1:
      readCache(self, j, true)
    var ii = self.indptr[j-self.offset]
    let iiMax = self.indptr[j-self.offset+1]
    while (ii < iiMax):
      yield (self.data[ii].field, self.data[ii].id, self.coef*self.data[ii].val)
      inc(ii)