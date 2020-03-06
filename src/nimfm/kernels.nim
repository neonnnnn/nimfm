import dataset, tensor, math


proc linear*(X: CSCDataset, w: Vector,  kernel: var seq[float64]) =
  let nFeatures = X.nFeatures
  for j in 0..<nFeatures:
    for (i, val) in X.getCol(j):
      kernel[i] += val * w[j]


proc linear*(X: CSRDataset, w: Vector, kernel: var seq[float64]) =
  let nSamples = X.nSamples
  for i in 0..<nSamples:
    for (j, val) in X.getRow(i):
      kernel[i] += w[j] * val


proc anova*(X: CSCDataset, P: Tensor, A: var Matrix,  
            degree, order, s: int, nAugments: int = 0) =
  let nSamples = X.nSamples
  let nFeatures = X.nFeatures
  for i in 0..<nSamples:
    for t in 1..<degree+1:
      A[i, t] = 0.0
    A[i, 0] = 1.0

  if degree != 2:
    for j in 0..<nFeatures:
      for (i, val) in X.getCol(j):
        for t in 0..<degree:
          A[i, degree-t] += A[i, degree-t-1] * P[order, s, j] * val
    # for augmented features
    for j in nFeatures..<(nFeatures+nAugments):
      for i in 0..<nSamples:
        for t in 0..<degree:
          A[i, degree-t] += A[i, degree-t-1] * P[order, s, j]
  else:
    for j in 0..<nFeatures:
      for (i, val) in X.getCol(j):
        A[i, 1] += P[order, s, j] * val
        A[i, 2] += (P[order, s, j]*val)^2
    for j in nFeatures..<(nFeatures+nAugments):
      for i in 0..<nSamples:
        A[i, 1] += P[order, s, j]
        A[i, 2] += P[order, s, j]^2
    for i in 0..<nSamples:
      A[i, 2] = (A[i, 1]^2 - A[i, 2])/2.0


proc anova*(X: CSRDataset, P: Tensor, A: var Matrix, 
            degree, order, s: int, nAugments:int = 0) =
  let nSamples = X.nSamples
  let nFeatures = X.nFeatures
  for i in 0..<nSamples:
    for t in 1..<degree+1:
      A[i, t] = 0.0
    A[i, 0] = 1.0
  
  if degree != 2:
    for i in 0..<nSamples:
      for (j, val) in X.getRow(i):
        for t in 0..<degree:
          A[i, degree-t] += A[i, degree-t-1] * P[order, s, j] * val
      for j in nFeatures..<(nFeatures+nAugments):
        for t in 0..<degree:
          A[i, degree-t] += A[i, degree-t-1] * P[order, s, j]
  else:
    for i in 0..<nSamples:
      for (j, val) in X.getRow(i):
        A[i, 2] += P[order, s, j] * val
        A[i, 1] += (P[order, s, j] * val)^2
      for j in nFeatures..<(nFeatures+nAugments):
        A[i, 2] += P[order, s, j]
        A[i, 1] += P[order, s, j]^2
      A[i, 2] = (A[i, 2]^2 - A[i, 1])/2.0
