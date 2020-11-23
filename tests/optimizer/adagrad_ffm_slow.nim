import nimfm/tensor/tensor, nimfm/loss
import sequtils, random
import ../model/ffm_slow
from adagrad_slow import AdaGradSlow, newAdaGradSlow, init, update
export AdaGradSlow, newAdaGradSlow


proc fit*[L](self: AdaGradSlow[L], X: Matrix, fields: seq[int],
             y: seq[float64], ffm: var FFMSlow) =
  ffm.init(X, fields)
  let y = ffm.checkTarget(y)
  let
    nSamples = X.shape[0]
    fitLinear = ffm.fitLinear
    fitIntercept = ffm.fitIntercept
    loss = self.loss
  var
    indices = toSeq(0..<nSamples)
    grad: Tensor = zeros(ffm.P.shape)

  init(self, ffm.P, ffm.w, ffm.warmStart, fitLinear, fitIntercept)

  for epoch in 0..<self.maxIter:
    if self.shuffle: shuffle(indices)
    for i in indices:
      # compute prediction
      let yPred = decisionFunction(ffm, X, fields, i)

      # compute gradient
      let dL = loss.dloss(y[i], yPred)
      grad <- 0.0
      computeGrad(ffm, X, fields, i, dL, grad)

      update(self, X, i, ffm.P, ffm.w, ffm.intercept, grad, dL,
             fitLinear, fitIntercept)
      
      inc(self.it)