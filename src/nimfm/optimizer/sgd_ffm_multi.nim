import ../dataset, ../tensor/tensor
import ../model/field_aware_factorization_machine
from ../model/fm_base import checkTarget
import sequtils, math, random, sugar, threadpool
from sgd import
  SGD, newSGD, finalize, stoppingCriterion, init, SchedulingKind
export sgd.SGD, sgd.newSGD, sgd.SchedulingKind
from sgd_multi import nThreads
from sgd_ffm import step

var 
  dA {.threadvar.}:  Tensor # zeros(P.shape)


proc epochSub[L](self: ptr SGD[L], X: ptr RowDataset, P: ptr Tensor,
                 w: ptr Vector, intercept: ptr float64,
                 y: ptr Vector, scaling_P, scaling_w: ptr float64,
                 scalings_P, scalings_w: ptr Vector,
                 nAugments: int, fitLinear, fitIntercept: bool,
                 indices: ptr seq[int], s, t: int): (float64, float64) =
  if dA.isNil or dA.shape != P[].shape:
    dA = zeros(P[].shape)
  for ii in s..<t:
    let i = indices[ii]
    step(self[], X[], P[], w[], intercept[], i, y[i], dA, scaling_P[],
         scaling_w[], scalings_P[], scalings_w[], nAugments,
         fit_linear, fitIntercept, result[0], result[1])
    inc(self[].it)


proc fit*[L](self: SGD[L], X: RowFieldDataset, y: seq[float64],
             ffm: FieldAwareFactorizationMachine, maxThreads: int, 
             callback: (SGD[L], FieldAwareFactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  ffm.init(X)

  let y = ffm.checkTarget(y)
  let
    nSamples = X.nSamples
    fitLinear = ffm.fitLinear
    fitIntercept = ffm.fitIntercept
    nThreads = nThreads(maxThreads)
  var
    scaling_w = 1.0
    scaling_P = 1.0
    scalings_w = ones([len(ffm.w)])
    scalings_P = ones([ffm.P.shape[1]])
    indices = toSeq(0..<nSamples)
    isConverged = false
    responses = newSeq[FlowVar[(float64, float64)]](nThreads)
    borders = newSeqWith(nThreads+1, 0)

  if not ffm.warmstart:
    self.init()
  
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
          addr(ffm.intercept), unsafeAddr(y), addr(scaling_P),addr(scaling_w),
          addr(scalings_P), addr(scalings_w), ffm.nAugments, fitLinear,
          fitIntercept, addr(indices), borders[th], borders[th+1])

      dec(nRest, X.nCached)
      for resp in responses:
        let ret = ^resp
        runningLoss += ret[0]
        viol += ret[1]

    # one epoch done
    runningLoss /= float(nSamples)
    if not callback.isNil:
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
  finalize(ffm.P, ffm.w, scaling_P, scaling_w, scalings_P, scalings_w, fitLinear)