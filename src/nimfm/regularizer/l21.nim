import ../tensor/tensor, ../dataset


type
  L21* = ref object
    norms: Vector
    value*: float64
    scaling: float64
    scalings: Vector
    threshold: float64
    thresholds: Vector


proc newL21*(): L21 = L21()


proc eval*(self: L21, P: Matrix): float64 =
  result = norm(norm(P, 2, axis=1), 1)


proc eval*(self: L21, P: Matrix, degree: int): float64 = self.eval(P)


# for pbcd
proc prox*(self: L21, pj: var Vector, lam: float64, degree, j: int) {.inline.} =
  let norm = norm(pj, 2)
  if norm > lam: pj *= (1.0 - lam / norm)
  else: 
    pj[0.. ^1] = 0.0


# for psgd/pgd/minibatch-psgd
proc prox*(self: L21, P: var Matrix, gamma: float64, degree: int) {.inline.} =
  for j in 0..<len(P):
    self.prox(P[j], gamma, degree, j)


proc initSGD*(self: L21, degree, nFeatures, nComponents: int) =
  self.scalings = ones([nFeatures])
  self.thresholds = zeros([nFeatures])
  self.scaling = 1.0
  self.threshold = 0.0


proc initBCD*(self: L21, degree, nFeatures, nComponents: int) =
  self.norms = zeros([nFeatures])


# for pbcd
# P.shape: (nFeatures, nComponents)
proc computeCacheBCD*(self: L21, P: Matrix, degree: int) =
  self.value = 0.0
  for j in 0..<P.shape[0]:
    self.norms[j] = norm(P[j], 2)
    self.value += self.norms[j]


proc updateCacheBCD*(self: L21, P: Matrix, degree, j: int) =
  self.value -= self.norms[j]
  self.norms[j] = norm(P[j], 2)
  self.value += self.norms[j]


proc lazyUpdate*(self: L21, P: var Matrix, beta, gamma: float64,
                 degree: int, X: RowDataset, i: int) {.inline.} =
  for (j, _) in X.getRow(i):
    let threshold = (self.threshold - self.thresholds[j]) / self.scalings[j]
    self.prox(P[j], threshold*gamma, degree, j)
    P[j] *= self.scaling / self.scalings[j] 


proc lazyUpdateFinal*(self:L21, P: var Tensor, beta, gamma: float64,
                      degree: int) {.inline.} =
  for order in 0..<P.shape[0]:
    for j in 0..<P.shape[1]:
      let threshold = (self.threshold - self.thresholds[j]) /  self.scalings[j]
      self.prox(P[order, j], threshold*gamma, degree, j)
      P[order, j] *= self.scaling / self.scalings[j] 


proc updateCacheSGD*(self: L21, eta, beta, gamma: float64,
                     degree: int, X: RowDataset, i: int) {.inline.} =
  self.threshold += eta * self.scaling
  self.scaling /= (1 + eta * beta)
  for (j, _) in X.getRow(i):
    self.scalings[j] = self.scaling
    self.thresholds[j] = self.threshold
  

proc resetCacheSGD*(self: L21, P: var Tensor, gamma: float64,
                    degree: int) {.inline.} = 
  if self.scaling < 1e-8: # to avoid numerical error
    for order in 0..<P.shape[0]:
      for j in 0..<P.shape[1]:
        let threshold = (self.threshold - self.thresholds[j]) / self.scalings[j]
        self.prox(P[order, j], threshold*gamma, degree, j)
        P[order, j] *= self.scaling / self.scalings[j] 
    self.threshold = 0.0
    self.scaling = 1.0
    self.thresholds[0.. ^1] = 0.0
    self.scalings[0.. ^1] = 1.0


# for psgd
proc step*(self: L21, P: var Matrix, dA: Matrix, 
           dL, beta, gamma, eta_P_scaled: float64, degree: int,
           indices: iterator) {.inline.} =
  let nComponents = P.shape[1]
  for j in indices():
    for s in 0..<nComponents:
      let update = eta_P_scaled * (dL*dA[j, s] + beta*P[j, s])
      P[j, s] -= update
    self.prox(P[j], eta_P_scaled*gamma, degree, j)