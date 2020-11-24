import ../dataset, ../tensor/tensor, ../loss
import ../model/field_aware_factorization_machine, ../model/params
from ../model/fm_base import checkTarget
from sgd import stoppingCriterion
from adagrad import AdaGrad, newAdaGrad, init, finalize, updateG, update
export adagrad.AdaGrad, adagrad.newAdaGrad
from sgd_ffm import predictWithGrad
from sgd_multi import nThreads
import sequtils, math, random, sugar, threadpool


var 
  dA {.threadvar.}:  Tensor # zeros(P.shape)


proc epochSub[L](self: ptr AdaGrad[L], X: ptr RowDataset, P: ptr Tensor,
                 w: ptr Vector, intercept: ptr float64, y: ptr Vector,
                 nAugments: int, fitLinear, fitIntercept: bool,
                 indices: ptr seq[int], s, t: int): (float64, float64) =
  if dA.isNil or dA.shape != P[].shape:
    dA = zeros(P[].shape)

  for ii in s..<t:
    let i = indices[ii]
    # update parameters lazily
    if self[].it != 1:
      result[1] += self[].update(P[], w[], intercept[], X[], i, nAugments, 
                                 fitLinear, fitIntercept)
    let yPred = predictWithGrad(X[], i, P[], w[], intercept[], dA)
    result[0] += self.loss.loss(y[i], yPred)
    updateG(self[], X[], dA, i, y[i], yPred, nAugments, fitLinear,
            fitIntercept)
    inc(self[].it)


proc fit*[L](self: AdaGrad[L], X: RowFieldDataset, y: seq[float64],
             ffm: FieldAwareFactorizationMachine, maxThreads: int,
             callback: (AdaGrad[L], FieldAwareFactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  ffm.init(X)

  let y = ffm.checkTarget(y)
  let
    nSamples = X.nSamples
    fitLinear = ffm.fitLinear
    fitIntercept = ffm.fitIntercept
    nThreads = nThreads(maxThreads)
  var
    indices = toSeq(0..<nSamples)
    isConverged = false
    responses = newSeq[FlowVar[(float64, float64)]](nThreads)
    borders = newSeqWith(nThreads+1, 0)
  
  # initialization
  init(self, ffm.P, ffm.w, ffm.warmStart, fitLinear, fitIntercept)
 
  for th in 0..<nThreads:
    borders[th+1] = borders[th] + nSamples div nThreads
  borders[^1] = nSamples


  for epoch in 0..<self.maxIter:
    var viol = 0.0
    var runningLoss = 0.0
    if X.nCached == X.nSamples and self.shuffle: shuffle(indices)
    
    var nRest = nSamples
    while nRest > 0:
      X.readCache(nSamples-nRest)
      borders[0] = nSamples - nRest
      for th in 0..<nThreads:
        borders[th+1] = borders[th] + X.nCached div nThreads
      borders[^1] = borders[0] + X.nCached
      
      # async parallel update!
      for th in 0..<nThreads:
        responses[th] = spawn epochSub(
          unsafeAddr(self), unsafeAddr(X), addr(ffm.P), addr(ffm.w),
          addr(ffm.intercept), unsafeAddr(y), ffm.nAugments, fitLinear,
          fitIntercept, addr(indices), borders[th], borders[th+1])

      dec(nRest, X.nCached)
      for resp in responses:
        let ret = ^resp
        runningLoss += ret[0]
        viol += ret[1]

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