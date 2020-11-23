import nimfm/tensor/tensor, nimfm/loss
import sequtils, random
import ../model/ffm_slow
from sgd_slow import SGDSlow, getEta, newSGDSlow
export SGDSlow, newSGDSlow


proc fit*[L](self: SGDSlow[L], X: Matrix, fields: seq[int], y: seq[float64],
             ffm: FFMSlow) =
  ffm.init(X, fields)
  let y = ffm.checkTarget(y)
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    alpha0 = self.alpha0
    alpha = self.alpha
    beta = self.beta
    fitLinear = ffm.fitLinear
    fitIntercept = ffm.fitIntercept
    loss = self.loss
  var
    indices = toSeq(0..<nSamples)
    grad: Tensor = zeros(ffm.P.shape)

  if not ffm.warmstart:
    self.it = 1
  for epoch in 0..<self.maxIter:
    var runningLoss = 0.0
    if self.shuffle: shuffle(indices)
    for i in indices:
      # compute prediction and gradient
      let yPred = decisionFunction(ffm, X, fields, i)
      runningLoss += loss.loss(y[i], yPred)
      
      # compute gradient
      let dL = loss.dloss(y[i], yPred)
      grad <- 0.0
      computeGrad(ffm, X, fields, i, dL, grad)

      # update parameters
      let wEta = self.getEta(alpha)
      let PEta = self.getEta(beta)
      if fitIntercept:
        let update = self.getEta(alpha0) * (dL + alpha0 * ffm.intercept)
        ffm.intercept -= update

      if fitLinear:
        for j in 0..<nFeatures:
          let update = wEta * (dL * X[i, j] + alpha * ffm.w[j])
          ffm.w[j] -= update  

      for f in 0..<ffm.P.shape[0]:
        for j in 0..<ffm.P.shape[1]:
          for s in 0..<ffm.P.shape[2]:
            let update =  PEta * (grad[f, j, s] + beta * ffm.P[f, j, s])
            ffm.P[f, j, s] -= update
      inc(self.it)
