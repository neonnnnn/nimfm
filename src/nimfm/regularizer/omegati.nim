import ../tensor/tensor, utils, math


type
  OmegaTI* = ref object
    dcache: Vector
    cache: Vector
    absp: Vector
    p: Vector
    value*: float64


proc newOmegaTI*(): OmegaTI =
  new(result)


proc eval*(self: OmegaTI, P: Matrix, degree: int): float64 =
  let nFeatures = P.shape[0]
  let nComponents = P.shape[1]
  var cache = zeros([degree+1, nComponents])
  cache[0, 0..^1] = 1.0
  for j in 0..<nFeatures:
    for deg in 0..<degree:
      for s in 0..<nComponents:
        cache[degree-deg, s] += self.cache[degree-deg-1]*abs(P[j, s])
  result = sum(cache[degree])


proc initCD*(self: OmegaTI, degree, nFeatures, nComponents: int) =
  self.dcache = zeros([degree+1])
  self.absp = zeros([nFeatures])
  self.cache = zeros([degree+1])


# for pcd
# P.shape: (nComponents, nFeatures)
proc computeCacheCDAll*(self: OmegaTI, P: Matrix, degree: int) = discard


proc computeCacheCD*(self: OmegaTI, P: Matrix, degree, s: int) =
  let nFeatures = P.shape[1]
  for j in 0..<nFeatures:
    self.absp[j] = abs(P[s, j])

  self.cache[1..^1] = 0.0
  self.cache[0] = 1.0
  self.dcache[0..^1] = 0.0
  self.dcache[1] = 1.0
  for j in 0..<nFeatures:
    for deg in 0..<degree:
      self.cache[degree-deg] += self.cache[degree-deg-1] * abs(P[s, j])
 

proc updateCacheCD*(self: OmegaTI, P: Matrix, degree, s, j: int) =
  for deg in 1..<degree:
    self.cache[deg] = self.dcache[deg+1] + self.dcache[deg] * abs(P[s, j])


# for pcd
proc prox*(self: OmegaTI, psj, update, lam: float64, 
           degree, s, j: int): float64 {.inline.} =
  for deg in 2..<degree+1: # dcache[deg] = derivative of cache[deg]
    self.dcache[deg] = self.cache[deg-1] - self.dcache[deg-1]*self.absp[j]
    if self.dcache[deg] < 0:
      self.dcache[deg] = 0.0
  result = softthreshold(psj-update, lam*self.dcache[degree])
