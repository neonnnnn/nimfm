import sequtils, random, math

type
  Vector* = ref object
    data: seq[float64]
    shape*: array[1, int]

  Matrix* = ref object
    data: seq[seq[float64]]
    shape*: array[2, int]

  Tensor* = ref object
    data: seq[Matrix]
    shape*: array[3, int]


proc len*[T: Tensor|Matrix|Vector](self: T): int =
  result = self.shape[0]


# for Vector subscription
proc `[]`*(self: Vector, i: int): var float64 {.inline.} =
  result = self.data[i]
proc `[]=`*(self: var Vector, i: int, val: float64) {.inline.} =
  self.data[i] = val
proc `[]+=`*(self: var Vector, i, j: int, val: float64) {.inline.} =
  self.data[i] += val
proc `[]-=`*(self: var Vector, i: int, val: float64) {.inline.} =
  self.data[i] -= val
proc `*=`(self: var Vector, val: float64) {.inline.} =
  for i in 0..<len(self):
    self.data[i] *= val
proc `/=`*(self: var Vector, val: float64) {.inline.} =
   self *= 1.0/val


# for Matrix subscription
proc `[]`*(self: Matrix, i, j: int): var float64 {.inline.} =
  result = self.data[i][j]
proc `[]`*(self: Matrix, i:int): var seq[float64] {.inline.} = self.data[i]
proc `[]=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] = val
proc `[]+=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] += val
proc `[]-=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] -= val
proc `*=`(self: var Matrix, val: float64) {.inline.} =
  for i in 0..<self.shape[0]:
    for j in 0..<self.shape[1]:
      self.data[i][j] *= val
proc `/=`*(self: var Matrix, val: float64) {.inline.} =
   self *= 1.0/val


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
proc `*=`(self: var Tensor, val: float64) {.inline.} =
  for i in 0..<self.shape[0]:
    self.data[i] *= val
proc `/=`*(self: var Tensor, val: float64) {.inline.} =
   self *= 1.0/val


proc toVector*(x: seq[float64]): Vector =
  new(result)
  result.shape = [len(x)]
  result.data = x


proc toMatrix*(X: seq[seq[float64]]): Matrix =
  new(result)
  result.shape = [len(X), len(X[0])]
  result.data = X


proc toTensor*(X: seq[seq[seq[float64]]]): Tensor =
  new(result)
  result.shape = [len(X), len(X[0]), len(X[0][0])]
  result.data = newSeq[Matrix](result.shape[0])
  for i, mat in X:
    result.data[i] = toMatrix(mat)


proc ones*(shape: array[2, int]): Matrix =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], newSeqWith(shape[1], 1.0))


proc ones*(shape: array[3, int]): Tensor =
  new(result)
  result.shape = shape
  result.data = newSeqWith(
    shape[0], ones([shape[1], shape[2]])
  )


proc ones*(shape: array[1, int]): Vector =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], 1.0)


proc zeros*(shape: array[2, int]): Matrix =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], newSeqWith(shape[1], 0.0))


proc zeros*(shape: array[1, int]): Vector =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], 0.0)


proc zeros*(shape: array[3, int]): Tensor =
  new(result)
  result.shape = shape
  result.data = newSeqWith(
    shape[0], zeros([shape[1], shape[2]])
  )


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


proc norm*(X: Vector, p: int): float64 =
  result = 0.0
  if p < 0:
    raise newException(ValueError, "p < 0.")
  for i in 0..<X.shape[0]:
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


proc add*(self: var Vector, val: float64) =
  self.data.add(val)
  self.shape[0] += 1


proc delete*(self: var Vector, i: int) = 
  self.data.delete(i, i)
  self.shape[0] -= 1


proc powerIteration*(X: Matrix, maxIter=20, tol=1e-4): 
                    tuple[value: float64, vector: Vector] = 
  let n = X.shape[0]
  if n != X.shape[1]:
    raise newException(ValueError, "X is not square matrix.")
  var vector = zeros([n])
  var cache = zeros([n])
  var ev = 0.0
  var evOld = 0.0
  # init
  for j in 0..<n:
    vector[j] = 2*rand(1.0) - 1.0
  vector /= norm(vector, 2)
  # start power iteration
  for it in 0..<maxIter:
    ev = 0.0
    # compute eigen value
    for i in 0..<n:
      var vi = 0.0
      for j in 0..<n:
        vi += X[i, j] * vector[j]
      ev += vector[i] * vi
      cache[i] = vi
    for i in 0..<n:
      vector[i] = cache[i]
    vector /= norm(vector, 2)
    if it > 0 and abs(ev-evOld) < tol:
      break
    evOld = ev
  result = (ev, vector)


when isMainModule:
  let X = toMatrix(@[@[8.0, 1.0], @[4.0, 5.0]])
  var eval: float64
  var evec: Vector
  for i in 0..<10:
    (eval, evec) = powerIteration(X, 10, 0)
    echo(eval)
    echo(evec[0], " ", evec[1])
    echo(norm(evec, 2))