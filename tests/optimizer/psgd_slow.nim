import nimfm/tensor/tensor, nimfm/loss
import nimfm/optimizer/optimizer_base, nimfm/optimizer/sgd
import sequtils, math, random
import ../model/fm_slow
import ../regularizer/squaredl12_slow

type
  PSGDSlow*[L, R] = ref object of BaseCSROptimizer
    gamma: float64
    loss: L
    reg: R
    eta0: float64
    scheduling: SchedulingKind
    power: float64
    it: int
    shuffle: bool


proc newPSGDSlow*[L, R](eta0 = 0.01, alpha0=1e-6, alpha=1e-3, beta=1e-4,
                        gamma=1e-4, loss: L = newSquared(), 
                        reg: R = newSquaredL12Slow(), scheduling = optimal,
                        power = 1.0, maxIter = 100, verbose = 1, tol = 1e-3,
                        shuffle = true): PSGDSlow[L, R] =
  result = PSGDSlow[L, R](eta0: eta0, alpha0: alpha0, alpha: alpha, beta: beta,
                          gamma: gamma, loss: loss, reg: reg,
                          scheduling: scheduling, power: power, it: 1, 
                          maxIter: maxIter, tol: tol, verbose: verbose,
                          shuffle: shuffle)


proc getEta(self: PSGDSlow, reg: float64): float64 {.inline.} =
  case self.scheduling
  of constant:
    result = self.eta0
  of optimal:
    result = self.eta0 / pow(1.0+self.eta0*reg*float(self.it), self.power)
  of invscaling:
    result = self.eta0 / pow(toFloat(self.it), self.power)
  of pegasos:
    result = 1.0 / (reg * toFloat(self.it))


proc fit*[L, R](self: PSGDSlow[L, R], X: Matrix, y: seq[float64],
                fm: var FMSlow) =
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
      let eta_w = self.getEta(alpha)
      let eta_P = self.getEta(beta)
      let eta_P_scaled = eta_P / (1.0 + eta_P * beta)
      if fitIntercept:
        let update = self.getEta(alpha0) * (dL + alpha0 * fm.intercept)
        fm.intercept -= update / (1.0 + self.getEta(alpha0)*alpha0)

      if fitLinear:
        for j in 0..<nFeatures:
          let update = eta_w * (dL * X[i, j] + alpha * fm.w[j])
          fm.w[j] -= update / (1.0 + eta_w * alpha)
    
      for order in 0..<nOrders:
        for s in 0..<nComponents:
          for j in 0..<(nFeatures+nAugments):
            let update =  (grad[order, s, j] + beta * fm.P[order, s, j])
            fm.P[order, s, j] -= eta_P_scaled * update
        self.reg.prox(fm.P[order], self.gamma*eta_P_scaled, degree-order)
      inc(self.it)