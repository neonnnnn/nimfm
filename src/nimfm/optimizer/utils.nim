import ../tensor/tensor, ../model/params, ../loss
import math, strformat, strutils


proc computeViol*(P, old_P: Tensor, w, old_w: Vector,
                  intercept, intercept_old: float64,
                  fitLinear, fitIntercept: bool): float64 {.inline.} =
  result = 0.0
  for order in 0..<P.shape[0]:
    for j in 0..<P.shape[1]:
      for s in 0..<P.shape[2]:
        result += (P[order, j, s] - old_P[order, j, s])^2
  if fitLinear:
    for j in 0..<len(w):
      result += (w[j] - old_w[j])^2
  if fitIntercept:
    result += (intercept - intercept_old)^2


proc computeViol*(params, old_params: Params): float64 =
  result = computeViol(params.P, old_params.P, params.w, old_params.w,
                       params.intercept, old_params.intercept,
                       params.fitLinear, params.fitIntercept)


proc echoHeader*(maxIter: int, viol=true, loss=true, regul=true) =
  let epoch = alignLeft("Epoch", len($maxIter))
  
  stdout.write(fmt"{epoch}")
  if viol:
    let viol = alignLeft("Violation", 10)
    stdout.write(fmt"   {viol}")
  if loss:
    let loss = alignLeft("Loss", 10)
    stdout.write(fmt"   {loss}")
  if regul:
    stdout.write(fmt"   Regularization")

  stdout.write("\n")
  stdout.flushFile()


proc echoInfo*(iter, maxIter: int, viol, loss, regul: float64) =
  let epoch = alignLeft($iter, max(5, len($maxIter)))
  stdout.write(fmt"{epoch}")
  if viol >= 0:
    stdout.write(fmt"   {viol:<10.4e}")
  if loss >= 0:
    stdout.write(fmt"   {loss:<10.4e}")
  if regul >= 0:
    stdout.write(fmt"   {regul:<10.4e}")
  stdout.write("\n")
  stdout.flushFile()


proc regularization*[T](P: T, w: Vector, intercept: float64,
                        alpha0, alpha, beta: float64): float64 =
  result = 0.5 * alpha0 * intercept^2 + 0.5 * alpha * norm(w, 2)^2
  result += 0.5 * beta * norm(P, 2)^2


proc regularization*(params: Params, alpha0, alpha, beta: float64): float64 =
  result = regularization(params.P, params.w, params.intercept,
                          alpha0, alpha, beta)


proc objective*[L, T](y: seq[float64], yPred: Vector, P: T, w: Vector,
                      intercept: float64, alpha0, alpha, beta: float64,
                      loss: L): (float64, float64) =
  result[0] = 0.0
  let nSamples = len(y)
  for i in 0..<nSamples:
    result[0] += loss.loss(y[i], yPred[i])
  result[0] /= float(nSamples)
  result[1] = regularization(P, w, intercept, alpha0, alpha, beta)


proc objective*[L](y: seq[float64], yPred: Vector, params: Params,
                    alpha0, alpha, beta: float64,
                    loss: L): (float64, float64) =
  result = objective(y, yPred, params.P, params.w, params.intercept,
                     alpha0, alpha, beta, loss)