import ../dataset, ../tensor/tensor, ../loss
import math


proc fitLinearCD*[L](w: var Vector, X: ColDataset, y: seq[float64],
                     yPred: var Vector, colNormSq: Vector,
                     alpha: float64, loss: L): float64 =
  result = 0.0
  let nFeatures = X.nFeatures
  var
    update = 0.0
    invStepSize = 0.0

  for j in 0..<nFeatures:
    update = alpha * w[j]
    for (i, val) in X.getCol(j):
      update += loss.dloss(y[i], yPred[i]) * val
    invStepSize = loss.mu * colNormSq[j] + alpha
    if invStepSize < 1e-12: 
      continue
    update /= invStepSize
    result += abs(update)
    w[j] -= update
    for (i, val) in X.getCol(j):
      yPred[i] -= update * val


proc fitInterceptCD*[L](intercept: var float64, y: seq[float64],
                        yPred: var Vector, nSamples: int,
                        alpha0: float64, loss: L): float64 =
  result = alpha0 * intercept
  for i in 0..<nSamples:
    result += loss.dloss(y[i], yPred[i])
  result /= loss.mu * float(nSamples) + alpha0
  intercept -= result
  for i in 0..<nSamples:
    yPred[i] -= result
  result = abs(result)


proc fitLinearSGD*(w: var Vector, X: RowDataset, i: int,
                   alpha, dL, eta: float64): float64 =
  result = 0.0
  for (j, val) in X.getRow(i):
    let update = eta * (dL * val + alpha*w[j])
    w[j] -= update
    result += abs(update)


proc fitLinearAdaGrad*(w, g_sum_w, g_norms_w: var Vector, X: RowDataset,
                       i: int, alpha, eta: float64, it: float64): float64 =
  result = 0.0
  let denom = it*eta*alpha
  for (j, val) in X.getRow(i):
    let wj = w[j]
    w[j] = - eta * g_sum_w[j] / (denom + sqrt(g_norms_w[j]))
    result += abs(wj - w[j])
