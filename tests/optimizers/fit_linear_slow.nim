import nimfm/loss, nimfm/tensor/tensor


proc fitLinearCD*[L](w: var Vector, X: Matrix, y: seq[float64],
                     yPred: var seq[float64], colNormSq: Vector,
                     alpha: float64, loss: L): float64 =
  result = 0.0
  let nSamples = X.shape[0]
  let nFeatures = X.shape[1]
  var
    update = 0.0
    invStepSize = 0.0

  for j in 0..<nFeatures:
    update = alpha * w[j]
    for i in 0..<nSamples:
      update += loss.dloss(y[i], yPred[i]) * X[i, j]
    invStepSize = loss.mu * colNormSq[j] + alpha
    update /= invStepSize
    result += abs(update)
    w[j] -= update
    for i in 0..<nSamples:
      yPred[i] -= update * X[i, j]


proc fitInterceptCD*[L](intercept: var float64, y: seq[float64],
                        yPred: var seq[float64], nSamples: int,
                        alpha0: float64, loss: L): float64 =
  result = alpha0 * intercept
  for i in 0..<nSamples:
    result += loss.dloss(y[i], yPred[i])
  result /= loss.mu * float(nSamples) + alpha0
  intercept -= result
  for i in 0..<nSamples:
    yPred[i] -= result
  result = abs(result)


proc fitLinearSGD*(w: var Vector, X: Matrix, alpha, dL, eta: float64,
                   i: int): float64 =
  result = 0.0
  for j in 0..<X.shape[1]:
    let update = eta * (dL * X[i, j] + alpha*w[j])
    w[j] -= update
    result += abs(update)
