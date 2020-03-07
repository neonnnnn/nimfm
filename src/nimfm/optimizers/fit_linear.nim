import ../loss, ../dataset, ../tensor


proc fitLinearCD*(w: var Vector, X: CSCDataset, y: seq[float64],
                  yPred: var seq[float64], colNormSq: Vector,
                  alpha: float64, loss: LossFunction): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  var
    update = 0.0
    invStepSize = 0.0

  for j in 0..<nFeatures:
    update = alpha * w[j]
    for (i, val) in X.getCol(j):
      update += loss.dloss(yPred[i], y[i]) * val
    invStepSize = loss.mu * colNormSq[j] + alpha
    if invStepSize < 1e-12: 
      continue
    update /= invStepSize
    result += abs(update)
    w[j] -= update
    for (i, val) in X.getCol(j):
      yPred[i] -= update * val


proc fitInterceptCD*(intercept: var float64, y: seq[float64],
                     yPred: var seq[float64], nSamples: int,
                     alpha0: float64, loss: LossFunction): float64 =
  result = alpha0 * intercept
  for i in 0..<nSamples:
    result += loss.dloss(yPred[i], y[i])
  result /= loss.mu * float(nSamples) + alpha0
  intercept -= result
  for i in 0..<nSamples:
    yPred[i] -= result
  result = abs(result)


proc fitLinearSGD*(w: var Vector, X: CSRDataset, alpha, dL, eta: float64,
                   i: int): float64 =
  result = 0.0
  for (j, val) in X.getRow(i):
    let update = eta * (dL * val + alpha*w[j])
    w[j] -= update
    result += abs(update)
