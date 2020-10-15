import nimfm/tensor/tensor
import algorithm, sequtils, utils, math, sugar


type
  SquaredL12Slow* = ref object
    transpose: bool


proc proxSquaredL12Slow*(p: var Vector, lam: float64, degree: int) {.inline.} =
  let n = len(p)
  var absp: seq[float64] = sorted(p.map(x=>abs(x)), order=Descending)
  var S = 2.0 * lam * cumsummed(absp)
  for i in 0..<n:
    S[i] /= (1.0 + 2.0*lam*(float64(i)+1.0))
  var theta: int = 0
  for i in 0..<n:
    if absp[i] - S[i] < 0: break
    inc(theta)
  for i in 0..<n:
    if abs(p[i]) < absp[theta-1]:
      p[i] = 0
    else:
      p[i] = softthreshold(p[i], S[theta-1])


# P.shape: (nFeatures, nComponents)
proc eval*(self: SquaredL12Slow, P: Matrix): float64 =
  ## Evaluates for P
  let axis = if self.transpose: 0 else: 1
  result = norm(norm(P, 1, axis=axis), 2)^2

  
proc newSquaredL12Slow*(transpose=true): SquaredL12Slow =
  ## Creates new SquaredL12Slow object.
  new(result)
  result.transpose = transpose


# for coordinate descent
# P.shape: [nComponents, nFeatures]
proc prox*(self: SquaredL12Slow, P: var Matrix, lam: float64, 
           degree, s, j: int) {.inline.} =
  var strength = 0.0
  let psj = P[s, j]
  if self.transpose: # input P is already transposed
    for j2 in 0..<P.shape[1]:
      strength += abs(P[s, j2])
  else:
    for s2 in 0..<P.shape[0]:
      strength += abs(P[s2, j])
  strength -= abs(psj)

  P[s, j] = softthreshold(psj / (1+2*lam), 2*lam*strength / (1+2*lam))


# for bcd
# P.shape: [nFeatures, nComponents]
proc prox*(self: SquaredL12Slow, P: var Matrix, lam: float64,
           degree, j: int) {.inline.} =
  if degree > 2:
    raise newException(ValueError, "degree > 2")
  if self.transpose:
    raise newException(ValueError, "transpose=true is not supported in PBCD.")
  else:
    proxSquaredL12Slow(P[j], lam, degree)


# for gd/sgd
# P.shape: [nComponents, nFeatures]
proc prox*(self: SquaredL12Slow, P: var Matrix, 
           lam: float64, degree: int) {.inline.} =
  if degree > 2:
    raise newException(ValueError, "degree > 2")
  
  if self.transpose:
    for s in 0..<P.shape[0]:
      proxSquaredL12Slow(P[s], lam, degree)
  else:
    var p = zeros([P.shape[0]])
    for j in 0..<P.shape[1]:
      for s in 0..<P.shape[0]:
        p[s] = P[s, j]
      proxSquaredL12Slow(p, lam, degree)
      for s in 0..<P.shape[0]:
        P[s, j] = p[s]
