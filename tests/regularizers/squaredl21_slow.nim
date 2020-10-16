import nimfm/tensor/tensor, squaredl12_slow
import math


type
  SquaredL21Slow* = ref object
    value*: float64
    cache: float64
    transpose: bool


proc newSquaredL21Slow*(transpose=false): SquaredL21Slow =
  new(result)
  result.transpose = transpose


# P.shape: (nFeatures, nComponents)
proc eval*(self: SquaredL21Slow, P: Matrix): float64 =
  let axis = if self.transpose: 0 else: 1
  result = norm(norm(P, 2, axis=axis), 1)^2


proc eval*(self: SquaredL21Slow, P: Matrix, degree: int): float64 =
  if degree != 2:
    raise newException(ValueError, "degree != 2.")
  let axis = if self.transpose: 0 else: 1
  result = norm(norm(P, 2, axis=axis), 1)^2


# for bcd
# P.shape: [nComponents, nFeatures]
# assume transpose = false
proc prox*(self: SquaredL21Slow, P: var Matrix, lam: float64,
           degree, j: int) {.inline.} =
  for s in 0..<P.shape[0]:
    P[s, j] /= (1+2*lam)
  let norms = norm(P, 2, axis=0)
  let lamScaled = 2.0 * lam / (1.0+2*lam) * (sum(norms) - norms[j])
  var shrink = 0.0
  if norms[j] > lamScaled: 
    shrink = 1.0 - lamScaled / norms[j]
  for s in 0..<P.shape[0]:
    P[s, j] *= shrink

# for gd/sgd
# P.shape: [nComponents, nFeatures]
proc prox*(self: SquaredL21Slow, P: var Matrix, 
           lam: float64, degree: int) {.inline.} =
  if not self.transpose:
    var norms = norm(P, 2, 0)
    for j in 0..<P.shape[1]:
      if norms[j] != 0.0:
        for s in 0..<P.shape[0]:
          P[s, j] /= norms[j]
    proxSquaredL12Slow(norms, lam, degree)
    for j in 0..<P.shape[1]:
      for s in 0..<P.shape[0]:
        P[s, j] *= norms[j]
  else:
    var norms = norm(P, 2, 1)
    for s in 0..<P.shape[0]:
      if norms[s] != 0:
        P[s] /= norms[s]
    proxSquaredL12Slow(norms, lam, degree)
    for s in 0..<P.shape[0]:
      P[s] *= norms[s]
