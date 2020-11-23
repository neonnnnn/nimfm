import ../dataset, ../tensor/tensor, ../loss
import ../model/field_aware_factorization_machine
from ../model/fm_base import checkTarget
import sequtils, math, random, sugar
from sgd import
  SGD, newSGD, lazilyUpdate, update, finalize, stoppingCriterion, init,
  SchedulingKind
export sgd.SGD, sgd.newSGD, sgd.SchedulingKind


proc predictWithGrad*(X: RowFieldDataset, i: int, P: Tensor, w: Vector,
                      intercept: float64, dA: var Tensor): float64 =
  result = intercept
  for (j, val) in X.getRow(i):
    result += w[j] * val

  # initialize dA
  for f in 0..<P.shape[0]:
    for j in X.getRowIndices(i):
      for s in 0..<P.shape[2]:
        dA[f, j, s] = 0.0 
  # compute prediction/gradient
  for (f1, j1, val1) in X.getRowWithField(i):
    for (f2, j2, val2) in X.getRowWithField(i):
      if j1 < j2:
        let tmp = dot(P[f2, j1], P[f1, j2])
        result += tmp * val1 * val2
        for s in 0..<P.shape[2]:
          dA[f2, j1, s] += val1 * val2 * P[f1, j2, s]
          dA[f1, j2, s] += val1 * val2 * P[f2, j1, s]


proc step*[L](self: SGD[L], X: RowDataset, P: var Tensor, w: var Vector,
              intercept: var float64, i: int, yi: float64, dA: var Tensor,
              scaling_P, scaling_w: var float64,
              scalings_P, scalings_w: var Vector, nAugments: int,
              fitLinear, fitIntercept: bool, runningLoss, viol: var float64) =
  # synchronize (lazily update) and compute prediction/gradient
  lazilyUpdate(X, i, P, w, scaling_P, scaling_w, scalings_P, scalings_w,
               fitLinear)
  let yPred = predictWithGrad(X, i, P, w, intercept, dA)
  runningLoss += self.loss.loss(yi, yPred)
  # nField * nComponents * nnz
  viol += update(self, X, P, w, intercept, dA, i, yi, yPred, scaling_P,
                 scaling_w, scalings_P, scalings_w, nAugments, fitLinear,
                 fitIntercept)


proc fit*[L](self: SGD[L], X: RowFieldDataset, y: seq[float64],
             ffm: FieldAwareFactorizationMachine,
             callback: (SGD[L], FieldAwareFactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  ffm.init(X)

  let y = ffm.checkTarget(y)
  let
    nSamples = X.nSamples
    fitLinear = ffm.fitLinear
    fitIntercept = ffm.fitIntercept
  var
    scaling_w = 1.0
    scaling_P = 1.0
    scalings_w = ones([len(ffm.w)])
    scalings_P = ones([ffm.P.shape[1]])
    indices = toSeq(0..<nSamples)
    dA: Tensor = zeros(ffm.P.shape)
    isConverged = false

  if not ffm.warmstart:
    self.init()

  for epoch in 0..<self.maxIter:
    var viol = 0.0
    var runningLoss = 0.0
    if X.nCached == X.nSamples and self.shuffle: shuffle(indices)

    for i in indices:
      step(self, X, ffm.P, ffm.w, ffm.intercept, i, y[i], dA, scaling_P,
           scaling_w, scalings_P, scalings_w, 0, fit_linear, fitIntercept,
           runningLoss, viol)

      if self.nCalls > 0 and self.it mod self.nCalls == 0:
        if not callback.isNil:
          finalize(ffm.P, ffm.w, scaling_P, scaling_w, scalings_P, scalings_w,
                   fitLinear)
          callback(self, ffm)
      inc(self.it)

    # one epoch done
    runningLoss /= float(nSamples)
    if not callback.isNil and self.nCalls <= 0:
      finalize(ffm.P, ffm.w, scaling_P, scaling_w, scalings_P, scalings_w,
               fitLinear)
      callback(self, ffm)
    let isContinue = stoppingCriterion(
      ffm.P, ffm.w, ffm.intercept, self.alpha0, self.alpha, self.beta,
      runningLoss, viol, self.tol, self.verbose, epoch, self.maxIter,
      isConverged)
    if not isContinue: break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(ffm.P, ffm.w, scaling_P, scaling_w, scalings_P, scalings_w,
           fitLinear)