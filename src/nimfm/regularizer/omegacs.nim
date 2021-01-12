import ../tensor/tensor


type
  OmegaCS* = ref object
    norms: Vector
    value*: float64
    dcache: Vector
    cache: Vector


proc newOmegaCS*(): OmegaCS = new(result)


proc eval*(self: OmegaCS, P: Matrix, degree: int): float64 =
  let nFeatures = P.shape[0]
  if len(self.norms) < nFeatures:
    self.norms.setLen(nFeatures)
  norm(P, self.norms, 2, axis=1)
 
  if len(self.cache) < degree+1:
    self.cache.setLen(degree+1)
  self.cache[0..^1] = 0.0
  self.cache[0] = 1.0
  for deg in 0..<degree: # compute (degree-1)-ANOVA kernel
    for j in 0..<nFeatures:
      self.cache[degree-deg] += self.cache[degree-deg-1]*self.norms[j]
  result = self.cache[degree]


proc initBCD*(self: OmegaCS, degree, nFeatures, nComponents: int) =
  self.norms = zeros([nFeatures])
  self.cache = zeros([degree+1])
  self.dcache = zeros([degree+1])
  self.dcache[1] = 1.0

# for pbcd
# P.shape: (nFeatures, nComponents)
proc recomputeCacheBCD(self: OmegaCS, degree: int) =
  self.cache[0.. ^1] = 0.0
  self.cache[0] = 1.0
  for j in 0..<len(self.norms):
    for deg in 0..<degree:
      self.cache[degree-deg] += self.cache[degree-deg-1] * self.norms[j]
  self.value = self.cache[degree]


proc computeCacheBCD*(self: OmegaCS, P: Matrix, degree: int) =
  for j in 0..<P.shape[0]:
    self.norms[j] = norm(P[j], 2)
  recomputeCacheBCD(self, degree)


proc updateCacheBCD*(self: OmegaCS, P: Matrix, degree, j: int) =
  let norm = norm(P[j], 2)
  for deg in 1..<degree+1:
    self.cache[deg] += self.dcache[deg] * norm 
    self.cache[deg] -= self.dcache[deg] * self.norms[j]
  self.norms[j] = norm
  if min(self.cache) < 0:
    recomputeCacheBCD(self, degree)
  self.value = self.cache[degree]


proc prox*(self: OmegaCS, pj: var Vector, lam: float64, degree, j: int) {.inline.} =
  let norm = norm(pj, 2)
  for deg in 2..<degree+1:
    self.dcache[deg] = self.cache[deg-1] - self.dcache[deg-1]*self.norms[j]

  # recompute since a numerical error occurs
  if min(self.dcache) < 0:
    self.norms[j] = 0.0
    recomputeCacheBCD(self, degree-1)
    self.dcache[0] = 0.0
    self.dcache[1] = 1.0
    for deg in 2..<degree+1:
      self.dcache[deg] = self.cache[deg-1]
    self.norms[j] = norm
    recomputeCacheBCD(self, degree)

  # prox!
  if norm > lam*self.dcache[degree]: 
    pj *= 1.0 - lam*self.dcache[degree] / norm
  else: 
    pj[0..^1] = 0.0
