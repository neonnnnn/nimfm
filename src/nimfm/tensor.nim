import sequtils, random, math, sugar, strformat
import nimlapack

# from system.nim
template `^^`(s, i: untyped): untyped =
  (when i is BackwardsIndex: s.len - int(i) else: int(i))


type
  Vector* = seq[float64]

  Matrix* = ref object
    data: seq[Vector]
    shape*: array[2, int]

  Tensor* = ref object
    data: seq[Matrix]
    shape*: array[3, int]


proc `$`*[T: Matrix|Tensor](X: T): string =
  result = "@[" & $X.data[0]
  for vec in X.data[1..^1]:
    result &= "\n  " & $vec
  result &= "]"


proc shape*[T](self: seq[T]): array[1, int] = [len(self)]


proc len*[T: Tensor|Matrix](self: T): int =
  result = self.shape[0]


# initialization
proc ones*(shape: array[1, int]): Vector =
  result = newSeqWith(shape[0], 1.0)


proc ones*(shape: array[2, int]): Matrix =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], ones([shape[1]]))


proc ones*(shape: array[3, int]): Tensor =
  new(result)
  result.shape = shape
  result.data = newSeqWith(
    shape[0], ones([shape[1], shape[2]])
  )


proc zeros*(shape: array[1, int]): Vector =
  result = newSeqWith(shape[0], 0.0)


proc zeros*(shape: array[2, int]): Matrix =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], zeros([shape[1]]))


proc zeros*(shape: array[3, int]): Tensor =
  new(result)
  result.shape = shape
  result.data = newSeqWith(
    shape[0], zeros([shape[1], shape[2]])
  )


# For Vector broadcasting
# Vector-scalar in-place operations
proc `+=`*[T](self: var seq[T], val: T) {.inline.} =
  for i in 0..<len(self):
    self[i] += val


proc `-=`*[T, U](self: var seq[T], val: T) {.inline.} =
  self += -val


proc `*=`*[T](self: var seq[T], val: T) {.inline.} =
  for i in 0..<len(self):
    self[i] *= val


proc `/=`*[T](self: var seq[T], val: T) {.inline.} =
  for i in 0..<len(self):
    self[i] /= val


proc `[]=`*[T, U, V](self: var seq[T], x: HSlice[U, V], val: T) {.inline.} =
  let a = self ^^ x.a
  let b = self ^^ x.b
  for i in a..b:
    self[i] = val


# seq[T]-scalar operations returning new seq[T]
proc `+`*[T](vec: seq[T], val: T): seq[T] =
  result = vec
  result += val


proc `-`*[T](vec: seq[T], val: T): seq[T] = vec + (-val)


proc `*`*[T](vec: seq[T], val: T): seq[T] =
  result = vec
  result *= val


proc `/`*[T](vec: seq[T], val: T): seq[T] =
  result = vec
  result /= val


proc `-`*[T](vec: seq[T]): seq[T] =
  result = vec * (-1)


proc `+`*[T](val: T, vec: seq[T]): seq[T] = vec + val


proc `-`*[T](val: T, vec: seq[T]): seq[T] = -(vec - val)


proc `*`*[T](val: T, vec: seq[T]): seq[T] = vec * val


proc `/`*[T](val: T, vec: seq[T]): seq[T] =
  result = vec
  for i, denom in result:
    result[i] = val / denom


# seq[T]-seq[T] in-place operations
proc `+=`*[T](self: var seq[T], vec: seq[T]) {.inline.} =
  if len(self) != len(vec):
    raise newException(ValueError, "len(self) != len(vec).")
  for i, val in vec:
    self[i] += val


proc `-=`*[T](self: var seq[T], vec: seq[T]) {.inline.} =
  if len(self) != len(vec):
    raise newException(ValueError, "len(self) != len(vec).")
  for i, val in vec:
    self[i] -= val


proc `*=`*[T](self: var seq[T], vec: seq[T]) {.inline.} =
  if len(self) != len(vec):
    raise newException(ValueError, "len(self) != len(vec).")
  for i, val in vec:
    self[i] *= val


proc `/=`*[T](self: var seq[T], vec: seq[T]) {.inline.} =
  if len(self) != len(vec):
    raise newException(ValueError, "len(self) != len(vec).")
  for i, val in vec:
    self[i] /= val


# seq[T]-seq[T] operations returning new seq[T]
proc `+`*[T](vec1, vec2: seq[T]): seq[T] =
  result = vec1
  result += vec2


proc `-`*[T](vec1, vec2: seq[T]): seq[T] =
  result = vec1
  result -= vec2


proc `*`*[T](vec1, vec2: seq[T]): seq[T] =
  result = vec1
  result *= vec2


proc `/`*[T](vec1, vec2: seq[T]): seq[T] =
  result = vec1
  result /= vec2


# for Matrix subscription
proc `[]`*(self: Matrix, i, j: int): var float64 {.inline.} =
  result = self.data[i][j]


proc `[]`*(self: Matrix, i: int): var Vector {.inline.} = self.data[i]


proc `[]=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] = val


proc `[]+=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] += val


proc `[]-=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] -= val


proc `[]=`*[U1, V1, U2, V2](self: var Matrix, x: HSlice[U1, V1],
                            y: HSlice[U2, V2], val: float64) {.inline.} =
  let a1 = self ^^ x.a
  let b1 = self ^^ x.b
  let a2 = self[0] ^^ y.a
  let b2 = self[0] ^^ y.b
  for i in a1..b1:
    for j in a2..b2:
      self[i, j] = val


# for Matrix broadcast
# Matrix-scalar/vector in-place operations
proc `+=`*[T: float64|Vector](self: var Matrix, val: T) {.inline.} =
  for i in 0..<len(self):
    self.data[i] += val


proc `-=`*[T: float64|Vector](self: var Matrix, val: T) {.inline.} = self += (-val)


proc `*=`*[T: float64|Vector](self: var Matrix, val: T) {.inline.} =
  for i in 0..<len(self):
    self.data[i] *= val


proc `/=`*[T: float64|Vector](self: var Matrix, val: T) {.inline.} =
  for i in 0..<len(self):
    self.data[i] /= val


# Matrix-scalar/vector operations returning new Matrix
proc `-`*(mat: Matrix): Matrix =
  new(result)
  deepCopy(result, mat)
  for i in 0..<len(result):
    result.data[i] *= -1


proc `+`*[T: float64|Vector](mat: Matrix, val: T): Matrix =
  new(result)
  deepCopy(result, mat)
  for i in 0..<len(result):
    result.data[i] += val


proc `-`*[T: float64|Vector](mat: Matrix, val: T): Matrix = mat + (-val)


proc `*`*[T: float64|Vector](mat: Matrix, val: T): Matrix = 
  new(result)
  deepCopy(result, mat)
  for i in 0..<len(result):
    result.data[i] *= val


proc `/`*[T: float64|Vector](mat: Matrix, val: T): Matrix =
  new(result)
  deepCopy(result, mat)
  for i in 0..<len(result):
    result.data[i] /= val


proc `+`*[T: float64|Vector](val: T, mat: Matrix): Matrix = mat + val


proc `-`*[T: float64|Vector](val: T, mat: Matrix): Matrix =
  new(result)
  deepCopy(result, mat)
  result *= -1
  result += val


proc `*`*[T: float64|Vector](val: T, mat: Matrix): Matrix = mat * val


proc `/`*[T: float64|Vector](val: T, mat: Matrix): Matrix =
  new(result)
  deepCopy(result, mat)
  for i in 0..<result.shape[0]:
    for j in 0..<result.shape[1]:
      result[i, j] = val / result[i, j]


proc transpose*(mat: Matrix): Matrix =
  new(result)
  result.shape = [mat.shape[1], mat.shape[0]]
  result.data = newSeqWith(mat.shape[1], zeros([mat.shape[0]]))
  for i in 0..<result.shape[0]:
    for j in 0..<result.shape[1]:
      result[i, j] = mat[j, i]


proc T*(mat: Matrix): Matrix = transpose(mat)


# for Tensor subsucription
proc `[]`*(self: Tensor, i, j, k: int): var float64 {.inline.} =
  result = self.data[i].data[j][k]


proc `[]`*(self: Tensor, i: int): var Matrix {.inline.} = self.data[i]


proc `[]=`*(self: var Tensor, i, j, k: int, val: float64) {.inline.} =
  self.data[i].data[j][k] = val


proc `[]+=`*(self: var Tensor, i, j, k: int, val: float64) {.inline.} =
  self.data[i].data[j][k] += val


proc `[]-=`*(self: var Tensor, i, j, k: int, val: float64) {.inline.} =
  self.data[i][j, k] -= val


proc toMatrix*(X: seq[seq[float64]]): Matrix =
  new(result)
  result.shape = [len(X), len(X[0])]
  result.data = newSeq[Vector](len(X))
  for i, vec in X:
    result.data[i] = vec


proc toTensor*(X: seq[seq[seq[float64]]]): Tensor =
  new(result)
  result.shape = [len(X), len(X[0]), len(X[0][0])]
  result.data = newSeq[Matrix](result.shape[0])
  for i, mat in X:
    result.data[i] = toMatrix(mat)


proc randomNormal*(shape: array[3, int], loc = 0.0, scale = 1.0): Tensor =
  # Box-Muller
  new(result)
  result = zeros(shape)
  var
    x, y, z: float64
    hasGauss: bool = false
  for i in 0..<shape[0]:
    for j in 0..<shape[1]:
      for k in 0..<shape[2]:
        if not hasGauss:
          x = rand(1.0)
          y = rand(1.0)
          z = sqrt(-2 * ln(1.0-x)) * cos(2*PI*y)
          hasGauss = true
        else:
          z = sqrt(-2 * ln(1.0-x)) * sin(2*PI*y)
          hasGauss = false

        result[i, j, k] = loc + z * scale


proc norm*(X: Tensor, p: int): float64 =
  result = 0.0
  if p < 0:
    raise newException(ValueError, "p < 0.")
  for i in 0..<X.shape[0]:
    for j in 0..<X.shape[1]:
      for k in 0..<X.shape[2]:
        if p == 0: result += float(X[i, j, k] != 0)
        else: result += abs(X[i, j, k])^p
  result = pow(result, 1.0/float(p))


proc norm*(X: Matrix, p: int): float64 =
  result = 0.0
  if p < 0:
    raise newException(ValueError, "p < 0.")
  for i in 0..<X.shape[0]:
    for j in 0..<X.shape[1]:
      if p == 0: result += float(X[i, j] != 0)
      else: result += abs(X[i, j])^p
  result = pow(result, 1.0/float(p))


proc norm*[T](X: T, p: int): float64 =
  result = 0.0
  if p < 0:
    raise newException(ValueError, "p < 0.")
  for i in 0..<len(X):
    if p == 0: result += float(X[i] != 0)
    else: result += abs(X[i])^p
  result = pow(result, 1.0/float(p))


proc addRow*[Vec](self: var Matrix, x: Vec) =
  self.data.add(newSeqWith(self.shape[1], 0.0))
  for j in 0..<len(x):
    self.data[^1][j] = x[j]
  self.shape[0] += 1


proc addCol*[Vec](self: var Matrix, x: Vec) =
  for i in 0..<len(x):
    self.data[i].add(x[i])
  self.shape[1] += 1


proc deleteRow*(self: var Matrix, i: int) = 
  self.data.delete(i, i)
  self.shape[0] -= 1


proc sum*[T: Matrix|Tensor](X: T): float64 = sum(X.data)


proc prod*[T: Matrix|Tensor](X: T): float64 = prod(X.data)


proc dot(x, y: float64): float64 {.inline.} = x*y


proc dot*[T: Vector|Matrix|Tensor](X, Y: T): float64 {.inline.} =
  ## Computes dot product between two vectors/matrices/tensors
  if X.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  result = 0.0
  for i in 0..<len(X):
    result += dot(X[i], Y[i])


# element-wise operation between two matrices/tensors
# in-place operations
proc `+=`*[T: Matrix|Tensor](self: var T, Y: T) =
  ## Compute in-place row-wise/element-wise addition
  if self.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  for i in 0..<len(self):
    self.data[i] += Y.data[i]


proc `-=`*[T: Matrix|Tensor](self: var T, Y: T) =
  ## Compute in-place row-wise/element-wise subtraction
  if self.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  for i in 0..<len(self):
    self.data[i] -= Y.data[i]


proc `*=`*[T: Matrix|Tensor](self: var T, Y: T) =
  ## Compute in-place row-wise/element-wise product
  if self.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  for i in 0..<len(self):
    self.data[i] *= Y.data[i]


proc `/=`*[T: Matrix|Tensor](self: var T, Y: T) =
  ## Compute in-place row-wise/element-wise division
  if self.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  for i in 0..<len(self):
    self.data[i] /= Y.data[i]


# returning new matrix/tensor operations
proc `+`*[T: Matrix|Tensor](X, Y: T): T {.inline.} =
  ## Computes element-wise addition
  if X.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  new(result)
  result.shape = X.shape
  result.data.setLen(len(X))
  for i in 0..<len(X):
    result.data[i] = X.data[i] + Y.data[i]


proc `-`*[T: Matrix|Tensor](X, Y: T): T {.inline.} =
  ## Computes element-wise subtraction
  if X.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  new(result)
  result.shape = X.shape
  result.data.setLen(len(X))
  for i in 0..<len(X):
    result.data[i] = X.data[i] - Y.data[i]


proc `*`*[T: Matrix|Tensor](X, Y: T): T {.inline.} =
  ## Computes element-wise product
  if X.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  new(result)
  result.shape = X.shape
  result.data.setLen(len(X))
  for i in 0..<len(X):
    result.data[i] = X.data[i] * Y.data[i]


proc `/`*[T: Matrix|Tensor](X, Y: T): T {.inline.} =
  ## Computes element-wise division
  if X.shape != Y.shape:
    raise newException(ValueError, "shape(X) != shape(Y).")
  new(result)
  result.shape = X.shape
  result.data.setLen(len(X))
  for i in 0..<len(X):
    result.data[i] = X.data[i] / Y.data[i]


proc mvmul*(mat: Matrix, vec: Vector, result: var Vector) =
  ## Multpies (n, m) matrix and (m, ) vector
  ## and substitutes output into result
  if mat.shape[1] != len(vec):
    raise newException(ValueError, "mat.shape[1] != len(vec).")
  if len(result) != mat.shape[0]:
    raise newException(ValueError, "len(result) != mat.shape[0].")
  result[.. ^1] = 0.0
  for i, mati in mat.data:
    result[i] = dot(mati, vec)


proc mvmul*(mat: Matrix, vec: Vector): Vector =
  ## Multpies (n, m) matrix and (m, ) vector
  ## Returns new (n, ) vector
  if mat.shape[1] != len(vec):
    raise newException(ValueError, "mat.shape[1] != len(vec).")
  result = newSeqWith(len(mat), 0.0)
  for i, mati in mat.data:
    result[i] = dot(mati, vec)


proc vmmul*(vec: Vector, mat: Matrix, result: var Vector) =
  ## Multpies (n, ) vector and (n, m) matrix
  ## and substitutes output into result
  if mat.shape[0] != len(vec):
    raise newException(ValueError, "mat.shape[0] != len(vec).")
  if len(result) != mat.shape[1]:
    raise newException(ValueError, "len(result) != nat.shape[1].")
  result[.. ^1] = 0.0
  for i in 0..<len(vec):
    for j in 0..<mat.shape[1]:
      result[j] += vec[i] * mat[i, j]


proc vmmul*(vec: Vector, mat: Matrix): Vector =
  ## Multpies (n, ) vector and (n, m) matrix
  ## Returns (m, ) vector
  if mat.shape[0] != len(vec):
    raise newException(ValueError, "mat.shape[0] != len(vec).")
  result = newSeqWith(mat.shape[1], 0.0)
  vmmul(vec, mat, result)


# Naive and too slow
proc matmul*(mat1, mat2: Matrix, result: var Matrix) =
  ## Multpies mat1: (n, k) matrix and mat2: (k, m) matrix
  ## and substitutes output into result
  if mat1.shape[1] != mat2.shape[0]:
    raise newException(ValueError, "mat1.shape[1] != mat2.shape[0]")
  let
    n = mat1.shape[0]
    m = mat2.shape[1]
  if result.shape != [n, m]:
    raise newException(ValueError, fmt"out.shape != [{n}, {m}].")
  # initialization
  for i in 0..<n:
    for j in 0..<m:
      result[i, j] = 0.0
  for i in 0..<n:
    for k in 0..<mat1.shape[1]:
      for j in 0..<m:
        result[i, j] += mat1[i, k] * mat2[k, j]


# Naive and too slow
proc matmul*(mat1, mat2: Matrix): Matrix =
  ## Multpies mat1: (n, k) matrix and mat2: (k, m) matrix
  ## Returns new (n, m) matrix
  if mat1.shape[1] != mat2.shape[0]:
    raise newException(ValueError, "mat1.shape[1] != mat2.shape[0]")
  let
    n = mat1.shape[0]
    m = mat2.shape[1]
  new(result)
  result.shape = [n, m]
  result.data = newSeqWith(n, zeros([m]))
  matmul(mat1, mat2, result)


# Modified Gram-Schmidt
proc orthogonalize*(X: var Matrix, n: int) =
  ## Orthogonalizes rows in X.
  for i in 0..<n:
    for j in 0..<i:
      var dot = 0.0
      for k in 0..<X.shape[1]:
        dot += X[i, k] * X[j, k]
      for k in 0..<X.shape[1]:
        X[i, k] -= dot * X[j, k]
    X[i] /= norm(X[i], 2)


proc powerMethod*(X: Matrix, maxIter=100, tol=1e-4): 
                  tuple[value: float64, vector: Vector] =
  ## Computes the dominate eigenvalue and corresponding eigenvector of X
  let n = X.shape[0]
  if n != X.shape[1]:
    raise newException(ValueError, "X is not square matrix.")
  var evec = zeros([n])
  var Xevec = zeros([n])
  var eval = 0.0
  var evalOld = 0.0
  # init
  for j in 0..<n:
    evec[j] = 2*rand(1.0) - 1.0
  evec /= norm(evec, 2)
  # start power iteration
  for it in 0..<maxIter:
    # compute eigen value
    mvmul(X, evec, Xevec)
    eval = dot(Xevec, evec)
    for i in 0..<n:
      evec[i] = Xevec[i]
    evec /= norm(evec, 2)
    if it > 0 and abs(eval-evalOld) < tol:
      break
    evalOld = eval
  result = (eval, evec)


proc powerMethod*(linear: (Vector, var Vector)->void, n:int, maxIter=100,
                  tol=1e-4): tuple[value: float64, vector: Vector] =
  ## Computes the dominate eigenvalue and corresponding eigenvector of X
  var evec = zeros([n])
  var Xevec = zeros([n])
  var eval = 0.0
  var evalOld = 0.0
  # init
  for j in 0..<n:
    evec[j] = 2*rand(1.0) - 1.0
  evec /= norm(evec, 2)
  # start power iteration
  for it in 0..<maxIter:
    # compute eigen value
    linear(evec, Xevec)
    eval = dot(evec, Xevec)
    for i in 0..<n:
      evec[i] = Xevec[i]
    evec /= norm(evec, 2)
    if it > 0 and abs(eval-evalOld) < tol:
      break
    evalOld = eval
  result = (eval, evec)


proc dsyev*(X: var Matrix, eigs: var Vector, columnWise=true) =
  ## Computes eigendecomposition of X
  if X.shape[0] != X.shape[1]:
    raise newException(ValueError, "X is not squared matrix.")
  let 
    jobs: cstring = "V"
    uplo: cstring = "U"
    n: int32 = int32(X.shape[0])
    info: int32 = 0
    lda: int32 = n
  var
    lwork: int32 = -1
    work: seq[float64] = @[0.0]
  # Rows of X might not be allocated continuously
  var A = newSeqWith(prod(X.shape), 0.0)
  # Fortran order
  for i in 0..<n:
    for j in i..<n:
      A[i+j*n] = X[i, j]
  # get optimal work size
  dsyev(jobs, uplo, n.unsafeAddr, A[0].unsafeAddr, lda.unsafeAddr,
        eigs[0].unsafeAddr, work[0].addr, lwork.addr, info.unsafeAddr)
  lwork = int32(work[0])
  work.setLen(lwork)
  # eigen decomposition
  dsyev(jobs, uplo, n.unsafeAddr, A[0].unsafeAddr, lda.unsafeAddr,
        eigs[0].addr, work[0].addr, lwork.addr, info.unsafeAddr)
  for i in 0..<n:
    for j in 0..<n:
      if columnWise: X[i, j] = A[i+j*n]
      else: X[j, i] = A[i+j*n] 


proc cg*(A: Matrix, b: Vector, x: var Vector, maxIter=1000, tol=1e-4) =
  if A.shape[0] != A.shape[1]:
    raise newException(ValueError, "A.shape[0] != A.shape[1].")
  let n = len(A)
  x[.. ^1] = 0.0
  var r = b
  var p = b
  var Ap = zeros([n])
  for it in 0..<maxIter:
    mvmul(A, p, Ap)
    let alpha = dot(r, p) / dot(p, Ap)
    p *= alpha
    x += p
    Ap *= alpha
    let norm = dot(r, r)
    let normNew = norm + dot(Ap, Ap) - 2 * dot(r, Ap)
    if normNew < tol: break
    let beta = normNew / norm
    r -= Ap
    p *= beta / alpha
    p += r


proc cg*(linear: (Vector, var Vector)->void, b: Vector, 
         x: var Vector, maxIter=1000, tol=1e-4) =
  let n = len(b)
  x[.. ^1] = 0.0
  var r = b
  var p = b
  var Ap = zeros([n])
  for it in 0..<maxIter:
    linear(p, Ap)
    let alpha = dot(r, p) / dot(p, Ap)
    p *= alpha
    x += p
    Ap *= alpha
    let norm = dot(r, r)
    let normNew = norm + dot(Ap, Ap) - 2 * dot(r, Ap)
    if normNew < tol: break
    let beta = normNew / norm
    r -= Ap
    p *= beta / alpha
    p += r


proc cgnr*(A: Matrix, b: Vector, x: var Vector, maxIter=1000, tol=1e-4) =
  let n = len(A)
  let m = len(x)
  x[.. ^1] = 0.0
  var r = vmmul(b, A) # len(r) = m
  var p = r
  var Ap = zeros([n]) 
  var AAp = zeros([m]) # A^\top A p
  for it in 0..<maxIter:
    mvmul(A, p, Ap)
    let alpha = dot(r, p) / dot(Ap, Ap)
    p *= alpha
    x += p
    vmmul(Ap, A, AAp)
    AAp *= alpha
    let norm = dot(r, r)
    let normNew = norm + dot(AAp, AAp) - 2 * dot(r, AAp)
    if normNew < tol: break
    let beta = normNew / norm
    r -= AAp
    p *= beta / alpha
    p += r


when isMainModule:
  var A = @[@[1.0, 1.0], @[1.0, 4.0]].toMatrix
  var b = @[3.0, 9.0]
  var x = @[0.0, 0.0]
  cg(A, b, x)

  var A2 = @[@[1.0, -2.0, 5.0], @[-0.3, 1.1, 0.5], @[0.4, -0.5, -0.2], @[2.0, 3.0, 0.1]].toMatrix
  var x2 = @[2.0, -1.0, 4.0]
  var b2 = mvmul(A2, x2)
  cgnr(A2, b2, x2)
  echo(x2)

  proc linear(p: Vector, x: var Vector) =
    vmmul(mvmul(A2, p), A2, x)
  cg(linear, vmmul(b2, A2), x2)
  echo(x2)