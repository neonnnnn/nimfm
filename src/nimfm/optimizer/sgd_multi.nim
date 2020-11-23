import ../dataset, ../tensor/tensor, ../model/factorization_machine
from ../model/fm_base import checkTarget
import sequtils, math, random, sugar, threadpool, cpuinfo
from sgd import
  SGD, init, stoppingCriterion, transpose, finalize, step, SchedulingKind
export sgd.SGD, sgd.newSGD, sgd.SchedulingKind

var 
  A {.threadvar.}: Matrix # zeros([nComponents, degree+1])
  dA {.threadvar.}:  Tensor # zeros(P.shape)


proc nThreads*(maxThreads: int):int =
  if maxThreads < 0:
    result = countProcessors() * 2
  else:
    result = maxThreads
  result = min(result, MaxThreadPoolSize)


proc epochSub[L](self: ptr SGD[L], X: ptr RowDataset, P: ptr Tensor,
                 w: ptr Vector, intercept: ptr float64,
                 y: ptr Vector, scaling_P, scaling_w: ptr float64,
                 scalings_P, scalings_w: ptr Vector,
                 nComponents, degree, nAugments: int,
                 fitLinear, fitIntercept: bool,
                 indices: ptr seq[int], s, t: int): (float64, float64) =
  if A.isNil or A.shape != [nComponents, degree+1]:
    A = zeros([nComponents, degree+1])
  if dA.isNil or dA.shape != P[].shape:
    dA = zeros(P[].shape)
  for ii in s..<t:
    let i = indices[ii]
    step(self[], X[], P[], w[], intercept[], i, y[i], A, dA, scaling_P[],
         scaling_w[], scalings_P[], scalings_w[], degree, nAugments,
         fit_linear, fitIntercept, result[0], result[1])
    inc(self[].it)


proc fit*[L](self: SGD[L], X: RowDataset, y: seq[float64],
             fm: FactorizationMachine, maxThreads: int,
             callback: (SGD[L], FactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  fm.init(X)

  let y = fm.checkTarget(y)
  let
    nSamples = X.nSamples
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    nAugments = fm.nAugments
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nThreads = nThreads(maxThreads)
  var
    scaling_w = 1.0
    scaling_P = 1.0
    scalings_w = ones([len(fm.w)])
    scalings_P = ones([fm.P.shape[2]])
    indices = toSeq(0..<nSamples)
    P: Tensor = zeros([nOrders, fm.P.shape[2], nComponents])
    isConverged = false
    responses = newSeq[FlowVar[(float64, float64)]](nThreads)
    borders = newSeqWith(nThreads+1, 0)

  if not fm.warmstart:
    self.init()
  
  for th in 0..<nThreads:
    borders[th+1] = borders[th] + nSamples div nThreads
  borders[^1] = nSamples

  # copy for fast training
  transpose(P, fm.P)

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
        responses[th] = spawn epochSub(unsafeAddr(self), unsafeAddr(X), addr(P),
          addr(fm.w), addr(fm.intercept), unsafeAddr(y), addr(scaling_P), addr(scaling_w),
          addr(scalings_P), addr(scalings_w), nComponents, degree, nAugments,
          fitLinear, fitIntercept, addr(indices), borders[th], borders[th+1])

      dec(nRest, X.nCached)
      for resp in responses:
        let ret = ^resp
        runningLoss += ret[0]
        viol += ret[1]

    # one epoch done
    runningLoss /= float(nSamples)
    if not callback.isNil:
      finalize(P, fm.w, scaling_P, scaling_w, scalings_P, scalings_w,
               fitLinear)
      transpose(fm.P, P)
      callback(self, fm)

    let isContinue = stoppingCriterion(
      P, fm.w, fm.intercept, self.alpha0, self.alpha, self.beta, runningLoss,
      viol, self.tol, self.verbose, epoch, self.maxIter, isConverged)
    if not isContinue: break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")
  # finalize
  finalize(P, fm.w, scaling_P, scaling_w, scalings_P, scalings_w, fitLinear)
  transpose(fm.P, P)