import sequtils, random, math

type
  Tensor* = ref object
    data: seq[seq[seq[float64]]]
    shape*: array[3, int]

  Matrix* = ref object
    data: seq[seq[float64]]
    shape*: array[2, int]

  Vector* = ref object
    data: seq[float64]
    shape*: array[1, int]


proc len*[T: Tensor, Vector, Matrix](self: T): int =
  result = self.shape[0]


# for Tensor subsucription
proc `[]`*(self: Tensor, i, j, k: int): var float64 {.inline.} =
  result = self.data[i][j][k]
proc `[]=`*(self: var Tensor, i, j, k: int, val: float64) {.inline.} =
  self.data[i][j][k] = val
proc `[]+=`*(self: var Tensor, i, j, k: int, val: float64) {.inline.} =
  self.data[i][j][k] += val
proc `[]-=`*(self: var Tensor, i, j, k: int, val: float64) {.inline.} =
  self.data[i][j][k] -= val


# for Matrix subscription
proc `[]`*(self: Matrix, i, j: int): var float64 {.inline.} =
  result = self.data[i][j]
proc `[]=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] = val
proc `[]+=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] += val
proc `[]-=`*(self: var Matrix, i, j: int, val: float64) {.inline.} =
  self.data[i][j] -= val


# for Vector subscription
proc `[]`*(self: Vector, i: int): var float64 {.inline.} =
  result = self.data[i]
proc `[]=`*(self: var Vector, i: int, val: float64) {.inline.} =
  self.data[i] = val
proc `[]+=`*(self: var Vector, i, j: int, val: float64) {.inline.} =
  self.data[i] += val
proc `[]-=`*(self: var Vector, i: int, val: float64) {.inline.} =
  self.data[i] -= val


proc toTensor*(X: seq[seq[seq[float64]]]): Tensor =
  new(result)
  result.shape = [len(X), len(X[0]), len(X[0][0])]
  result.data = X
  return result


proc toMatrix*(X: seq[seq[float64]]): Matrix =
  new(result)
  result.shape = [len(X), len(X[0])]
  result.data = X
  return result


proc zeros*(shape: array[1, int]): Vector =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], 0.0)


proc zeros*(shape: array[2, int]): Matrix =
  new(result)
  result.shape = shape
  result.data = newSeqWith(shape[0], newSeqWith(shape[1], 0.0))


proc zeros*(shape: array[3, int]): Tensor =
  new(result)
  result.shape = shape
  result.data = newSeqWith(
    shape[0], newSeqWith(shape[1], newSeqWith(shape[2], 0.0))
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
