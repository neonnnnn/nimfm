import nimfm/tensor/tensor
import math
import ../kernels_slow

type
  OmegaCSSlow* = ref object
    norms: Vector
    value*: float64


proc newOmegaCSSlow*(): OmegaCSSlow =
  new(result)


# P.shape: (nFeatures, nComponents)
proc eval*(self: OmegaCSSlow, P: Matrix, degree: int): float64 =
  var norms = zeros([P.shape[0]])
  for j in 0..<P.shape[0]:
    for pjs in P[j]:
      norms[j] += pjs^2
    norms[j] = sqrt(norms[j])
  let ones = ones([len(norms)])
  result = anovaSlow(norms, ones, degree)


# for prox BCD
# P.shape: [nComponents, nFeatures]
proc prox*(self: OmegaCSSlow, P: var Matrix, lam: float64, degree, j: int) {.inline.} =
  var norms = zeros([P.shape[1]])
  for s in 0..<P.shape[0]:
    for j1 in 0..<P.shape[1]:
      norms[j1] += P[s, j1]^2
  
  for j1 in 0..<P.shape[1]:
    norms[j1] = sqrt(norms[j1])

  let norm = norms[j]
  norms[j] = 0.0
  let ones = ones([len(norms)])
  var strength = lam*anovaSlow(norms, ones, degree-1)
  var shrink = 0.0
  if norm > strength:
    shrink = 1.0 - strength / norm
  for s in 0..<P.shape[0]:
    P[s, j] *= shrink