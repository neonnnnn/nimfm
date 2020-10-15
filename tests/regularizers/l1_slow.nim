import nimfm/tensor/tensor, utils


type
  L1Slow* = ref object
    scaling: float64
    scalings: Vector
    threshold: float64
    thresholds: Vector
    norms: Vector
    value*: float64


proc newL1Slow*(): L1Slow = L1Slow()

# P.shape: (nFeatures, nComponents)
proc eval*(self: L1Slow, P: Matrix): float64 =
  result = 0.0
  for j in 0..<P.shape[0]:
    for s in 0..<P.shape[1]:
      result += abs(P[j, s])


proc eval*(self: L1Slow, P: Matrix, degree: int): float64 = self.eval(P)


# for pcd
proc prox*(self: L1Slow, P: var Matrix, lam: float64,
           degree, s, j: int) {.inline} =
  P[s, j] = softthreshold(P[s, j], lam)

# for bcd
# P.shape = [nComponents, nFeatures]
proc prox*(self: L1Slow, P: var Matrix, lam: float64, degree, j: int) {.inline.} =
  for s in 0..<P.shape[0]:
    P[s, j] = softthreshold(P[s, j], lam)


# for psgd/pgd
# P.shape = [nComponents, nFeatures]
proc prox*(self: L1Slow, P: var Matrix, lam: float64, degree: int) {.inline.} =
  for s in 0..<len(P):
    for j in 0..<P.shape[1]:
      P[s, j] = softthreshold(P[s, j], lam)
