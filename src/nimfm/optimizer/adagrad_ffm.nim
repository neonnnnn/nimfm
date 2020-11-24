import ../dataset, ../tensor/tensor, ../loss
import ../model/field_aware_factorization_machine, ../model/params
from ../model/fm_base import checkTarget
from sgd import stoppingCriterion
from sgd_ffm import predictWithGrad
from adagrad import AdaGrad, newAdaGrad, update, updateG, init, finalize
import sequtils, math, random, sugar
export adagrad.AdaGrad, adagrad.newAdaGrad


proc fit*[L](self: AdaGrad[L], X: RowFieldDataset, y: seq[float64],
             ffm: FieldAwareFactorizationMachine,
             callback: (AdaGrad[L], FieldAwareFactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  ffm.init(X)

  let y = ffm.checkTarget(y)
  let
    nSamples = X.nSamples
    fitLinear = ffm.fitLinear
    fitIntercept = ffm.fitIntercept
  var
    indices = toSeq(0..<nSamples)
    df: Tensor = zeros(ffm.P.shape)
    isConverged = false
  
  # initialization
  init(self, ffm.P, ffm.w, ffm.warmStart, fitLinear, fitIntercept)

  for epoch in 0..<self.maxIter:
    var viol = 0.0
    var runningLoss = 0.0
    if X.nCached == X.nSamples and self.shuffle: shuffle(indices)
    
    for i in indices:
      # update parameters lazily
      if self.it != 1:
        viol += update(self, ffm.P, ffm.w, ffm.intercept, X, i, ffm.nAugments,
                       fitLinear, fitIntercept)
      let yPred = predictWithGrad(X, i, ffm.P, ffm.w, ffm.intercept, df)
      runningLoss += self.loss.loss(y[i], yPred)
      updateG(self, X, df, i, y[i], yPred, X.nAugments, fitLinear, fitIntercept)

      if self.nCalls > 0 and self.it mod self.nCalls == 0:
        if not callback.isNil:
          finalize(self, ffm.P, ffm.w, ffm.intercept, fitLinear, fitIntercept)
          callback(self, ffm)
      inc(self.it)

    # one epoch done
    runningLoss /= float(nSamples)
    if not callback.isNil:
      finalize(self, ffm.P, ffm.w, ffm.intercept, fitLinear, fitIntercept)
      callback(self, ffm)

    let isContinue = stoppingCriterion(
      ffm.P, ffm.w, ffm.intercept, self.alpha0, self.alpha, self.beta,
      runningLoss, viol, self.tol, self.verbose, epoch, self.maxIter,
      isConverged)
    if not isContinue: break
    
  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")
  
  # finalize
  finalize(self, ffm.P, ffm.w, ffm.intercept, fitLinear, fitIntercept)