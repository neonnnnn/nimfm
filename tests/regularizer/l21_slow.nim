import nimfm/tensor/tensor
import math

type
  L21Slow* = ref object
    norms: Vector
    value*: float64


proc newL21Slow*(): L21Slow = L21Slow()


# P.shape: (nFeatures, nComponents)
proc eval*(self: L21Slow, P: Matrix): float64 =
  result = 0.0
  for j in 0..<P.shape[0]:
    var norm = 0.0
    for s in 0..<P.shape[1]:
      norm += P[j, s]^2
    result += sqrt(norm)


proc eval*(self: L21Slow, P: Matrix, degree: int): float64 = self.eval(P)


# for bcd
# P.shape: [nComponents, nFeatures]
proc prox*(self: L21Slow, P: var Matrix, lam: float64, degree, j: int) {.inline.} =
  var norm = 0.0
  for s in 0..<P.shape[0]:
    norm += P[s, j]^2
  norm = sqrt(norm)
  var shrink = 0.0
  if norm > lam:
    shrink = 1.0 - lam / norm
  for s in 0..<P.shape[0]:
    P[s, j] *= shrink


# for sgd/gd
# P.shape: [nComponents, nFeatures]
proc prox*(self: L21Slow, P: var Matrix, gamma: float64, degree: int) {.inline.} =
  for j in 0..<P.shape[1]:
    self.prox(P, gamma, degree, j)
