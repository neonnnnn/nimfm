import ../tensor/tensor, ../dataset
from squaredl12 import proxSquaredL12
import sequtils, math


type
  SquaredL21* = ref object
    norms: Vector
    candidates: seq[int]
    value*: float64
    cache: float64
    transpose: bool


proc newSquaredL21*(transpose=false): SquaredL21 =
  new(result)
  result.transpose = transpose


proc eval*(self: SquaredL21, P: Matrix): float64 =
  let axis = if self.transpose: 0 else: 1
  result = norm(norm(P, 2, axis=axis), 1)^2


proc eval*(self: SquaredL21, P: Matrix, degree: int): float64 =
  if degree != 2:
    raise newException(ValueError, "SquaredL21 supports only degree=2.")
  result = self.eval(P)


# for bcd
proc prox*(self: SquaredL21, pj: var Vector, lam: float64,
           degree, j: int) {.inline.} =
  for s in 0..<len(pj):
    pj[s] /= (1+2*lam)
  let norm = norm(pj, 2)
  if self.cache < self.norms[j]:
    self.cache = sum(self.norms)
  let lamScaled = 2.0 * lam / (1.0+2*lam) * (self.cache - self.norms[j])
  if norm > lamScaled: 
    pj *= 1.0 - lamScaled / norm
  else: 
    pj[.. ^1] = 0.0


# for pgd/psgd/minibatch-psgd
# P.shape: [nFeatures, nComponents]
proc prox*(self: SquaredL21, P: var Matrix, 
           lam: float64, degree: int) {.inline.} =
  if not self.transpose:
    norm(P, self.norms, 2, 1)
    for i in 0..<P.shape[0]:
      if self.norms[i] != 0:
        P[i] /= self.norms[i]
    proxSquaredL12(self.norms, lam, self.candidates)
    for i in 0..<P.shape[0]:
      P[i] *= self.norms[i]
  else:
    norm(P, self.norms, 2, 0)
    for j in 0..<P.shape[0]:
      if self.norms[j] != 0.0:
        for s in 0..<P.shape[1]:
          P[j, s] /= self.norms[j]
    proxSquaredL12(self.norms, lam, self.candidates)
    P *= self.norms


proc initBCD*(self: SquaredL21, degree, nFeatures, nComponents: int) =
  if degree != 2:
    raise newException(ValueError, "SquaredL21 supports only degree=2.")
  if self.transpose:
    raise newException(ValueError, "transpose=true is not supported for BCD.")
  self.norms = zeros([nFeatures])


proc initSGD*(self: SquaredL21, degree, nFeatures, nComponents: int) =
  ## Initializes SquaredL21 object for PSGD solver.
  if degree != 2:
    raise newException(ValueError, "SquaredL12 supports only degree=2.")

  if self.transpose:
    self.candidates = newSeqWith(nComponents, 0)
    self.norms = zeros([nComponents])
  else:
    self.candidates = newSeqWith(nFeatures, 0)
    self.norms = zeros([nComponents])

# for pbcd
# P.shape: (nFeatures, nComponents)
proc computeCacheBCD*(self: SquaredL21, P: Matrix, degree: int,
                      indices: seq[int]) =
  for j in 0..<P.shape[0]:
    self.norms[j] = norm(P[j], 2)
  self.cache = sum(self.norms)
  self.value = self.cache^2


proc updateCacheBCD*(self: SquaredL21, P: Matrix, degree, j: int) =
  self.cache -= self.norms[j]
  self.norms[j] = norm(P[j], 2)
  self.cache += self.norms[j]
  self.value = self.cache^2

# for psgd
proc lazyUpdate*(self: SquaredL21, P: var Matrix, beta, gamma: float64,
                 degree: int, X: RowDataset, i: int) {.inline.} = discard
  

proc lazyUpdateFinal*(self:SquaredL21, P: var Tensor, 
                      beta, gamma: float64, degree: int) {.inline.} = discard
  

proc updateCacheSGD*(self: SquaredL21, eta, beta, gamma: float64,
                     degree: int, X: RowDataset, i: int) {.inline.} = discard


proc resetCacheSGD*(self: SquaredL21, P: var Tensor, 
                    gamma: float64, degree: int) {.inline.} = discard



proc step*(self: SquaredL21, P: var Matrix, dA: Matrix, 
           dL, beta, gamma, eta_P_scaled: float64, degree: int,
           indices: iterator) {.inline.} =
  # Updates all parameters (i.e., sparsity is not leveraged)
  for j in 0..<P.shape[0]:
    for s in 0..<P.shape[1]:
      P[j, s] -= eta_P_scaled * (dL*dA[j, s] + beta*P[j, s])

  self.prox(P, eta_P_scaled*gamma, degree)
