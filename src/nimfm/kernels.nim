import dataset, tensor, math


proc linear*[T, U](X: CSCDataset, w: T, kernel: var U) =
  let nFeatures = X.nFeatures
  let nSamples = X.nSamples
  for i in 0..<nSamples:
    kernel[i] = 0.0
  for j in 0..<nFeatures:
    for (i, val) in X.getCol(j):
      kernel[i] += val * w[j]


proc linear*[T, U](X: CSRDataset, w: T, kernel: var U) =
  let nSamples = X.nSamples
  for i in 0..<nSamples:
    kernel[i] = 0.0
    for (j, val) in X.getRow(i):
      kernel[i] += w[j] * val


proc anova*(X: CSCDataset, P: Matrix, A: var Matrix,  
            degree, s: int, nAugments: int = 0) =
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
          A[i, degree-t] += A[i, degree-t-1] * P[s, j] * val
    # for augmented features
    for j in nFeatures..<(nFeatures+nAugments):
      for i in 0..<nSamples:
        for t in 0..<degree:
          A[i, degree-t] += A[i, degree-t-1] * P[s, j]
  else:
    for j in 0..<nFeatures:
      for (i, val) in X.getCol(j):
        A[i, 1] += P[s, j] * val
        A[i, 2] += (P[s, j]*val)^2
    # for augmented features
    for j in nFeatures..<(nFeatures+nAugments):
      for i in 0..<nSamples:
        A[i, 1] += P[s, j]
        A[i, 2] += P[s, j]^2
    # finalize
    for i in 0..<nSamples:
      A[i, 2] = (A[i, 1]^2 - A[i, 2])/2.0


proc anova*(X: CSRDataset, P: Matrix, A: var Matrix, 
            degree, s: int, nAugments: int = 0) =
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
          A[i, degree-t] += A[i, degree-t-1] * P[s, j] * val
      # for augmented features
      for j in nFeatures..<(nFeatures+nAugments):
        for t in 0..<degree:
          A[i, degree-t] += A[i, degree-t-1] * P[s, j]
  else:
    for i in 0..<nSamples:
      for (j, val) in X.getRow(i):
        A[i, 2] += P[s, j] * val
        A[i, 1] += (P[s, j] * val)^2
      # for augmented features
      for j in nFeatures..<(nFeatures+nAugments):
        A[i, 2] += P[s, j]
        A[i, 1] += P[s, j]^2
      A[i, 2] = (A[i, 2]^2 - A[i, 1])/2.0


proc poly*(X: CSCDataset, P: Matrix, A: var Matrix,  
           degree, s: int, nAugments: int = 0) =
  let nSamples = X.nSamples
  let nFeatures = X.nFeatures
  for i in 0..<nSamples:
    A[i, 0] = 1.0

  for j in 0..<nFeatures:
    for (i, val) in X.getCol(j):
        A[i, 1] += P[s, j] * val

  # for augmented features
  for j in nFeatures..<(nFeatures+nAugments):
    for i in 0..<nSamples:
        A[i, 1] += P[s, j]
    
  for i in 0..<nSamples:
    for order in 2..<degree+1:
      A[i, order] = A[i, 1]^order


proc poly*(X: CSRDataset, P: Matrix, A: var Matrix, 
            degree, s: int, nAugments: int = 0) =
  let nSamples = X.nSamples
  let nFeatures = X.nFeatures
  for i in 0..<nSamples:
    A[i, 0] = 1.0
  
  for i in 0..<nSamples:
    for (j, val) in X.getRow(i):
      A[i, 1] += P[s, j] * val
    # for augmented features
    for j in nFeatures..<(nFeatures+nAugments):
      A[i, 1] += P[s, j]
  
  for i in 0..<nSamples:
    for order in 2..<degree+1:
      A[i, order] = A[i, 1]^order
