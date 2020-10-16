import ../tensor/tensor, ../dataset, utils


type
  L1* = ref object
    scaling: float64
    scalings: Vector
    threshold: float64
    thresholds: Vector
    norms: Vector
    value*: float64


proc newL1*(): L1 = L1()


# P.shape: (nFeatures, nComponents)
proc eval*(self: L1, P: Matrix): float64 = norm(P, 1)


proc eval*(self: L1, P: Matrix, degree: int): float64 = norm(P, 1)


# for pcd
proc prox*(self: L1, psj, update, lam: float64,
           degree, s, j: int): float64 {.inline} =
  result = softthreshold(psj-update, lam)


# for pbcd
proc prox*(self: L1, pj: var Vector, lam: float64, degree, j: int) {.inline.} =
  for i in 0..<len(pj):
    pj[i] = softthreshold(pj[i], lam)


# for psgd/pgd/minibatch-psgd
# P.shape: [nFeatures, nComponents]
proc prox*(self: L1, P: var Matrix, lam: float64, degree: int) {.inline.} =
  for j in 0..<len(P):
    for s in 0..<P.shape[1]:
      P[j, s] = softthreshold(P[j, s], lam)


proc initCD*(self: L1, degree, nFeatures, nComponents: int) = discard


proc initSGD*(self: L1, degree, nFeatures, nComponents: int) =
  self.scalings = ones([nFeatures])
  self.scaling = 1.0
  self.thresholds = zeros([nFeatures])
  self.threshold = 0.0


proc initBCD*(self: L1, degree, nFeatures, nComponents: int) =
  self.norms = zeros([nFeatures])

# for pcd
proc computeCacheCDAll*(self: L1, P: Matrix, degree: int) = discard
proc computeCacheCD*(self: L1, P: Matrix, degree, s: int) = discard
proc updateCacheCD*(self: L1, P: Matrix, degree, s, j: int) = discard


# for pbcd
# P.shape: (nFeatures, nComponents)
proc computeCacheBCD*(self: L1, P: Matrix, degree: int,
                      indices: seq[int]) =
  self.value = 0.0
  for j in 0..<P.shape[0]:
    self.norms[j] = norm(P[j], 1)
    self.value += self.norms[j]


proc updateCacheBCD*(self: L1, P: Matrix, degree, j: int) =
  self.value -= self.norms[j]
  self.norms[j] = norm(P[j], 1)
  self.value += self.norms[j]


# for psgd
# P.shape: (nFeatures, nComponents)
proc lazyUpdate*(self: L1, P: var Matrix, beta, gamma: float64,
                 degree: int, X: RowDataset, i: int) {.inline.} =
  for (j, _) in X.getRow(i):
    for s in 0..<P.shape[1]:
      P[j, s] *= self.scaling / self.scalings[j]
      P[j, s] = softthreshold(
        P[j, s], gamma * self.scaling * (self.threshold - self.thresholds[j]))


proc lazyUpdateFinal*(self:L1, P: var Tensor, beta, gamma: float64,
                      degree: int) {.inline.} =
  for order in 0..<P.shape[0]:
    for j in 0..<P.shape[1]:
      for s in 0..<P.shape[2]:
        P[order, j, s] *= self.scaling / self.scalings[j]
        P[order, j, s] = softthreshold(
          P[order, j, s], gamma * self.scaling * (self.threshold - self.thresholds[j]))
  self.scalings[0..^1] = 1.0
  self.scaling = 1.0
  self.thresholds[0..^1] = 0.0
  self.threshold = 0.0


proc updateCacheSGD*(self: L1, eta, beta, gamma: float64,
                     degree: int, X: RowDataset, i: int) {.inline.} =
  let eta_P_scaled = eta / (1+eta*beta)
  self.threshold += eta_P_scaled / self.scaling
  self.scaling *= (1 - eta_P_scaled * beta)
  for (j, _) in X.getRow(i):
    self.scalings[j] = self.scaling
    self.thresholds[j] = self.threshold
  

proc resetCacheSGD*(self: L1, P: var Tensor, gamma: float64,
                    degree: int) {.inline.} = 
  if self.scaling < 1e-8:
    for order in 0..<P.shape[0]:
      for j in 0..<P.shape[1]:
        for s in 0..<P.shape[2]:
          P[order, j, s] /= self.scalings[j]
          P[order, j, s] = softthreshold(
            P[order, j, s], gamma * self.threshold - self.thresholds[j])
          P[order, j, s] *= self.threshold
    self.threshold = 0.0
    self.scaling = 1.0
    self.thresholds[0.. ^1] = 0.0
    self.scalings[0.. ^1] = 1.0


proc step*(self: L1, P: var Matrix, dA: Matrix, 
           dL, beta, gamma, eta_P_scaled: float64, degree: int,
           indices: iterator) {.inline.} =
  let nComponents = P.shape[1]
  for j in indices():
    for s in 0..<nComponents:
      let update = eta_P_scaled * (dL*dA[j, s] + beta*P[j, s])
      P[j, s] = softthreshold(P[j, s]-update, gamma*eta_P_scaled)
