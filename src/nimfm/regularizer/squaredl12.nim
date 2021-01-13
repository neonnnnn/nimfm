import ../tensor/tensor, ../dataset
import sequtils, random, utils, math


type
  SquaredL12* = ref object
    transpose: bool
    cache: Vector
    absp: Vector
    candidates: seq[int]
    p: Vector
    norms: Vector
    value*: float64


proc proxSquaredL12*(p: var Vector, lam: float64, 
                     candidates: var seq[int]) {.inline.} =
  let n = len(p)
  var 
    S = 0.0
    theta: int = 0
    offset = 0
    nCandidates = n
  if len(candidates) < n:
    candidates.setLen(n)

  for i in 0..<n:
    candidates[i] = i

  # find p_theta
  while nCandidates != 0:
    let 
      ii = rand(nCandidates-1)
      i = candidates[offset+ii]
      pivot = abs(p[i])
    # partition candidates to L and G
    # L uses candidates[offset..<offset+nL]
    # G uses candidates[offset+nL..<offset+nCandidates]
    swap(candidates[offset+ii], candidates[offset+nCandidates-1])
    var
      nG = 1
      nL = 0
      SGi = pivot
    for ii2 in 0..<nCandidates-1:
      let i2 = candidates[offset+ii2]
      if pivot > abs(p[i2]):
        swap(candidates[offset+nL], candidates[offset+ii2])
        inc(nL)
      else:
        inc(nG)
        SGi += abs(p[i2])
    # determine whether to use L or G in next search step
    if pivot > 2*lam*(S+SGi)/(1.0+2.0*lam*float64(theta+nG)): # L
      offset = offset
      nCandidates = nL
      S += SGi
      theta += nG
    else: # G
      offset = offset+nL
      nCandidates = 0
      for ii2 in 0..<nG-1:
        let i2 = candidates[offset+ii2]
        # use elements in G that are strictly bigger than the pivot value
        if pivot < abs(p[i2]):
          swap(candidates[offset+ii2], candidates[offset+nCandidates])
          inc(nCandidates)
  S /= 1.0 + 2.0*lam*float64(theta)
  for i in 0..<n:
    p[i] = softthreshold(p[i], 2*lam*S)


proc eval*(self: SquaredL12, P: Matrix): float64 =
  ## Evaluates for P
  let axis = if self.transpose: 0 else: 1
  result = norm(norm(P, 1, axis=axis), 2)^2


proc eval*(self: SquaredL12, P: Matrix, degree: int): float64 = 
  if degree > 2:
    raise newException(ValueError, "SquaredL12 supports only degree=2.")
  result = self.eval(P)

  
proc newSquaredL12*(transpose=true): SquaredL12 =
  ## Creates new SquaredL12 object.
  new(result)
  result.transpose = transpose


proc initCD*(self: SquaredL12, degree, nFeatures, nComponents: int) =
  ## Initializes SquaredL12 object for PCD solver.
  if degree != 2:
    raise newException(ValueError, "SquaredL12 supports only degree=2.")

  self.absp = zeros([nFeatures])
  if self.transpose:
    self.cache = zeros([1])
  else:
    self.cache = zeros([nFeatures]) 


proc initSGD*(self: SquaredL12, degree, nFeatures, nComponents: int) =
  ## Initializes SquaredL12 object for PSGD solver.
  if degree != 2:
    raise newException(ValueError, "SquaredL12 supports only degree=2.")

  if self.transpose:
    self.candidates = newSeqWith(nFeatures, 0)
    self.p = zeros([nFeatures])
  else:
    self.candidates = newSeqWith(nComponents, 0)
  

proc initBCD*(self: SquaredL12, degree, nFeatures, nComponents: int) =
  ## Initializes SquaredL12 object for PBCD solver.
  if degree != 2:
    raise newException(ValueError, "SquaredL12 supports only degree=2.")

  if self.transpose:
    raise newException(ValueError, "transpose=true is not supported for BCD.")

  self.norms = zeros([nFeatures])
  self.candidates = newSeqWith(2*nComponents, 0)


# for pcd
proc prox*(self: SquaredL12, psj, update, lam: float64, 
           degree, s, j: int): float64 {.inline.} =
  let i = if self.transpose: 0 else: j
  let dcache = self.cache[i] - self.absp[j]
  result = softthreshold((psj-update) / (1+2*lam), 2*lam*dcache / (1+2*lam))


# for pbcd
proc prox*(self: SquaredL12, p: var Vector, lam: float64,
           degree, j: int) {.inline.} =
  # j is not used.
  if degree != 2:
    raise newException(ValueError, "SquaredL12 supports only degree=2.")
  if self.transpose:
    raise newException(ValueError, "transpose=true is not supported in PBCD.")
  proxSquaredL12(p, lam, self.candidates)



# for pgd/psgd/minibatch-psgd
proc prox*(self: SquaredL12, P: var Matrix, 
           lam: float64, degree: int) {.inline.} =
  if self.transpose:
    if len(self.p) < P.shape[0]:
      self.p.setLen(P.shape[0])

    for s in 0..<P.shape[1]:
      for j in 0..<P.shape[0]:
        self.p[j] = P[j, s]
      proxSquaredL12(self.p, lam, self.candidates)
      for j in 0..<P.shape[0]:
        P[j, s] = self.p[j]
  else:
    for j in 0..<P.shape[0]:
      proxSquaredL12(P[j], lam, self.candidates)


# for pcd
# P.shape: (nComponents, nFeatures)
proc computeCacheCDAll*(self: SquaredL12, P: Matrix, degree: int) =
  if not self.transpose:
    let nComponents = P.shape[0]
    let nFeatures = P.shape[1]
    self.cache[0..^1] = 0.0
    for s in 0..<nComponents:
      for j in 0..<nFeatures:
        self.cache[j] += abs(P[s, j])


proc computeCacheCD*(self: SquaredL12, P: Matrix, degree, s: int) =
  let nFeatures = P.shape[1]
  for j in 0..<nFeatures:
    self.absp[j] = abs(P[s, j])

  if self.transpose:
    self.cache[0] = sum(self.absp)


proc updateCacheCD*(self: SquaredL12, P: Matrix, degree, s, j: int) =
  let i = if self.transpose: 0 else: j
  self.cache[i] -= self.absp[j]
  self.cache[i] += abs(P[s, j])
   

# for bcd with line search
# P.shape: (nFeatures, nComponents)
proc computeCacheBCD*(self: SquaredL12, P: Matrix, degree: int) =
  for j in 0..<P.shape[0]:
    self.norms[j] = norm(P[j], 1)
  self.value = norm(self.norms, 2)^2


proc updateCacheBCD*(self: SquaredL12, P: Matrix, degree, j: int) =
  self.value -= self.norms[j]^2
  self.norms[j] = norm(P[j], 1)
  self.value += self.norms[j]^2


# for psgd
# P.shape = (nFeatures, nComponents)
proc lazyUpdate*(self: SquaredL12, P: var Matrix, beta, gamma: float64,
                 degree: int, X: RowDataset, i: int) {.inline.} = discard
  

proc lazyUpdateFinal*(self:SquaredL12, P: var Tensor, 
                      beta, gamma: float64, degree: int) {.inline.} = discard
  

proc updateCacheSGD*(self: SquaredL12, eta, beta, gamma: float64,
                     degree: int, X: RowDataset, i: int) {.inline.} = discard


proc resetCacheSGD*(self: SquaredL12, P: var Tensor, 
                    gamma: float64, degree: int) {.inline.} = discard


proc step*(self: SquaredL12, P: var Matrix, dA: Matrix, 
           dL, beta, gamma, eta_P_scaled: float64, degree: int,
           indices: iterator) {.inline.} =
  # Updates all parameters (i.e., sparsity is not leveraged)
  for j in 0..<P.shape[0]:
    for s in 0..<P.shape[1]:
      P[j, s] -= eta_P_scaled * (dL*dA[j, s] + beta*P[j, s])
  
  self.prox(P, gamma*eta_P_scaled, degree)