import nimfm/tensor/tensor, nimfm/loss
import nimfm/optimizers/optimizer_base, nimfm/optimizers/sgd
import sequtils, math, random
import ../kernels_slow, ../models/fm_slow, ../comb

type
  SGDSlow*[L] = ref object of BaseCSROptimizer
    loss: L
    eta0: float64
    scheduling: SchedulingKind
    power: float64
    it: int
    shuffle: bool


proc newSGDSlow*[L](eta0 = 0.01, alpha0=1e-6, alpha=1e-3, beta=1e-3,
                    loss: L = newSquared(),scheduling = optimal, power = 1.0,
                    maxIter = 100, verbose = 1, tol = 1e-3,
                    shuffle = true): SGDSlow[L] =
  result = SGDSlow[L](eta0: eta0, alpha0: alpha0, alpha: alpha, beta: beta,
                      loss: loss, scheduling: scheduling, power: power, it: 1,
                      maxIter: maxIter, tol: tol, verbose: verbose,
                      shuffle: shuffle)


proc getEta(self: SGDSlow, reg: float64): float64 {.inline.} =
  case self.scheduling
  of constant:
    result = self.eta0
  of optimal:
    result = self.eta0 / pow(1.0+self.eta0*reg*float(self.it), self.power)
  of invscaling:
    result = self.eta0 / pow(toFloat(self.it), self.power)
  of pegasos:
    result = 1.0 / (reg * toFloat(self.it))


# compute derivatives naively
# dA/dpj =  anova_without_j  * xj
proc computeDerivatives*(P: Tensor, X: Matrix, dA: var Tensor,
                         i, degree, order, nAugments: int) =
  let
    nFeatures = X.shape[1]
    nComponents = P.shape[1]
  
  for s in 0..<nComponents:
    for j in 0..<(nFeatures+nAugments):
      dA[order, s, j] = 0
      for indices in combNotj(nFeatures+nAugments, degree-1, j):
        var prod = 1.0
        for j2 in indices:
          prod *= P[order, s, j2]
          if j2 < nFeatures:
            prod *= X[i, j2]
        dA[order, s, j] += prod
      if j < nFeatures:
        dA[order, s, j] *= X[i, j]
  

proc computeAnovaGrad*(P: Tensor, X: Matrix, i, degree, order, nAugments: int,
                       dA: var Tensor): float64 =
  result = 0.0
  let
    nComponents = P.shape[1]
    nFeatures = X.shape[1]
  # compute anova kernel
  for s in 0..<nComponents:
    result += anovaSlow(X, P[order], i, degree, s, nFeatures, nAugments)

  # compute derivatives
  computeDerivatives(P, X, dA, i, degree, order, nAugments)


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
    dA: Tensor = zeros(fm.P.shape)

  if not fm.warmstart:
    self.it = 1
  for epoch in 0..<self.maxIter:
    var runningLoss = 0.0
    if self.shuffle: shuffle(indices)
    for i in indices:
      # compute prediction and gradient
      var yPred = fm.intercept
      for j in 0..<nFeatures:
        yPred += fm.w[j] * X[i, j]

      for order in 0..<nOrders:
        yPred += computeAnovaGrad(fm.P, X, i, degree-order, order, nAugments,
                                  dA)
      runningLoss += loss.loss(y[i], yPred)
      let dL = loss.dloss(y[i], yPred)

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
            let update =  PEta * (dL * dA[order, s, j] + beta * fm.P[order, s, j])
            fm.P[order, s, j] -= update

      inc(self.it)