import nimfm/tensor/tensor, nimfm/loss
import nimfm/optimizer/optimizer_base, nimfm/optimizer/sgd
import sequtils, math, random
import ../kernels_slow, ../model/fm_slow, ../comb

type
  SGDSlow*[L] = ref object of BaseCSROptimizer
    loss*: L
    eta0*: float64
    scheduling: SchedulingKind
    power: float64
    it*: int
    shuffle*: bool


proc newSGDSlow*[L](eta0 = 0.01, alpha0=1e-6, alpha=1e-3, beta=1e-3,
                    loss: L = newSquared(),scheduling = optimal, power = 1.0,
                    maxIter = 100, verbose = 1, tol = 1e-3,
                    shuffle = true): SGDSlow[L] =
  result = SGDSlow[L](eta0: eta0, alpha0: alpha0, alpha: alpha, beta: beta,
                      loss: loss, scheduling: scheduling, power: power, it: 1,
                      maxIter: maxIter, tol: tol, verbose: verbose,
                      shuffle: shuffle)


proc getEta*(self: SGDSlow, reg: float64): float64 {.inline.} =
  case self.scheduling
  of constant:
    result = self.eta0
  of optimal:
    result = self.eta0 / pow(1.0+self.eta0*reg*float(self.it), self.power)
  of invscaling:
    result = self.eta0 / pow(toFloat(self.it), self.power)
  of pegasos:
    result = 1.0 / (reg * toFloat(self.it))


proc fit*[L](self: SGDSlow[L], X: Matrix, y: seq[float64], fm: var FMSlow) =
  fm.init(X)
  let y = fm.checkTarget(y)
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    alpha0 = self.alpha0
    alpha = self.alpha
    beta = self.beta
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nAugments = fm.nAugments
    loss = self.loss
  var
    indices = toSeq(0..<nSamples)
    grad: Tensor = zeros(fm.P.shape)

  if not fm.warmstart:
    self.it = 1
  for epoch in 0..<self.maxIter:
    var runningLoss = 0.0
    if self.shuffle: shuffle(indices)
    for i in indices:
      # compute prediction
      let yPred = decisionFunction(fm, X, i)
      runningLoss += loss.loss(y[i], yPred)
      
      # compute gradient
      let dL = loss.dloss(y[i], yPred)
      grad <- 0.0
      computeGrad(fm, X, i, dL, grad)

      # update parameters
      let wEta = self.getEta(alpha)
      let PEta = self.getEta(beta)
      if fitIntercept:
        let update = self.getEta(alpha0) * (dL + alpha0 * fm.intercept)
        fm.intercept -= update

      if fitLinear:
        for j in 0..<nFeatures:
          let update = wEta * (dL * X[i, j] + alpha * fm.w[j])
          fm.w[j] -= update  
    
      for order in 0..<nOrders:
        for s in 0..<nComponents:
          for j in 0..<(nFeatures+nAugments):
            let update =  PEta * (grad[order, s, j] + beta * fm.P[order, s, j])
            fm.P[order, s, j] -= update

      inc(self.it)