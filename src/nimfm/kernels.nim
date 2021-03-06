import dataset, tensor/tensor, math


proc linear*[T, U](X: ColDataset, w: T, kernel: var U) =
  let nFeatures = X.nFeatures
  let nSamples = X.nSamples
  for i in 0..<nSamples:
    kernel[i] = 0.0
  for j in 0..<nFeatures:
    for (i, val) in X.getCol(j):
      kernel[i] += val * w[j]


proc linear*[T, U](X: RowDataset, w: T, kernel: var U) =
  let nSamples = X.nSamples
  for i in 0..<nSamples:
    kernel[i] = 0.0
    for (j, val) in X.getRow(i):
      kernel[i] += w[j] * val


proc anova*(X: ColDataset, P: Matrix, A: var Matrix,  
            degree, s: int) =
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
  else:
    for j in 0..<nFeatures:
      for (i, val) in X.getCol(j):
        A[i, 1] += P[s, j] * val
        A[i, 2] += (P[s, j]*val)^2
    # finalize
    for i in 0..<nSamples:
      A[i, 2] = (A[i, 1]^2 - A[i, 2])/2.0


proc anova*(X: RowDataset, P: Matrix, A: var Matrix, 
            degree, s: int) =
  let nSamples = X.nSamples
  for i in 0..<nSamples:
    for t in 1..<degree+1:
      A[i, t] = 0.0
    A[i, 0] = 1.0
  
  if degree != 2:
    for i in 0..<nSamples:
      for (j, val) in X.getRow(i):
        for t in 0..<degree:
          A[i, degree-t] += A[i, degree-t-1] * P[s, j] * val
  else:
    for i in 0..<nSamples:
      for (j, val) in X.getRow(i):
        A[i, 1] += P[s, j] * val
        A[i, 2] += (P[s, j] * val)^2
      A[i, 2] = (A[i, 1]^2 - A[i, 2])/2.0


proc poly*(X: ColDataset, P: Matrix, A: var Matrix,  
           degree, s: int) =
  let nSamples = X.nSamples
  let nFeatures = X.nFeatures
  A[0..^1, 1..^1] = 0.0
  A[0..^1, 0] = 1.0
  for j in 0..<nFeatures:
    for (i, val) in X.getCol(j):
        A[i, 1] += P[s, j] * val

  for i in 0..<nSamples:
    for order in 2..<degree+1:
      A[i, order] = A[i, order-1]*A[i, 1]


proc poly*(X: RowDataset, P: Matrix, A: var Matrix, 
            degree, s: int) =
  let nSamples = X.nSamples

  A[0..^1, 0..^1] = 0.0
  A[0..^1, 0] = 1.0

  for i in 0..<nSamples:
    for (j, val) in X.getRow(i):
      A[i, 1] += P[s, j] * val
  
  for i in 0..<nSamples:
    for order in 2..<degree+1:
      A[i, order] = A[i, order-1]*A[i, 1]
