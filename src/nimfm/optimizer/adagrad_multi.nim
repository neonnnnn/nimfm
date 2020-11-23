import ../dataset, ../tensor/tensor, ../model/factorization_machine
import ../model/params, ../loss
from ../model/fm_base import checkTarget
from sgd import predictWithGrad, stoppingCriterion, transpose
from sgd_multi import nThreads
from adagrad import AdaGrad, newAdaGrad, updateG, update, finalize, init
export adagrad.AdaGrad, adagrad.newAdaGrad
import sequtils, math, random, sugar, threadpool

var 
  A {.threadvar.}: Matrix # zeros([nComponents, degree+1])
  dA {.threadvar.}:  Tensor # zeros(P.shape)


proc epochSub[L](self: ptr AdaGrad[L], X: ptr RowDataset, P: ptr Tensor,
                 w: ptr Vector, intercept: ptr float64,
                 y: ptr Vector, nComponents, degree, nAugments: int,
                 fitLinear, fitIntercept: bool,
                 indices: ptr seq[int], s, t: int): (float64, float64) =
  if A.isNil or A.shape != [nComponents, degree+1]:
    A = zeros([nComponents, degree+1])
  if dA.isNil or dA.shape != P[].shape:
    dA = zeros(P[].shape)

  for ii in s..<t:
    let i = indices[ii]
    # update parameters lazily
    if self[].it != 1:
      result[1] += self[].update(P[], w[], intercept[], X[], i, nAugments, 
                                 fitLinear, fitIntercept)
    let yPred = predictWithGrad(X[], i, P[], w[], intercept[], A, dA, degree,
                                nAugments)
    result[0] += self.loss.loss(y[i], yPred)
    updateG(self[], X[], dA, i, y[i], yPred, nAugments, fitLinear,
            fitIntercept)
    inc(self[].it)


proc fit*[L](self: AdaGrad[L], X: RowDataset, y: seq[float64],
             fm: FactorizationMachine, maxThreads: int,
             callback: (AdaGrad[L], FactorizationMachine)->void = nil) =
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
    indices = toSeq(0..<nSamples)
    P: Tensor = zeros([nOrders, fm.P.shape[2], nComponents])
    isConverged = false
    responses = newSeq[FlowVar[(float64, float64)]](nThreads)
    borders = newSeqWith(nThreads+1, 0)
  
  # initialization
  init(self, P, fm.w, fm.warmStart, fitLinear, fitIntercept)
 
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
          addr(fm.w), addr(fm.intercept), unsafeAddr(y), nComponents, degree,
          nAugments, fitLinear, fitIntercept, addr(indices), borders[th],
          borders[th+1])

      dec(nRest, X.nCached)
      for resp in responses:
        let ret = ^resp
        runningLoss += ret[0]
        viol += ret[1]

    # one epoch done
    runningLoss /= float(nSamples)
    if not callback.isNil:
      finalize(self, P, fm.w, fm.intercept, fitLinear, fitIntercept)
      transpose(fm.P, P)
      callback(self, fm)
    
    let isContinue = stoppingCriterion(
      P, fm.w, fm.intercept, self.alpha0, self.alpha, self.beta, runningLoss,
      viol, self.tol, self.verbose, epoch, self.maxIter, isConverged)
    if not isContinue: break

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")
  
  # finalize
  finalize(self, P, fm.w, fm.intercept, fitLinear, fitIntercept)
  transpose(fm.P, P)