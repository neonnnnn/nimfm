import nimfm/tensor/tensor, nimfm/loss
import nimfm/optimizers/optimizer_base
import sequtils, math, random
import ../models/fm_slow
from sgd_slow import computeAnovaGrad, computeDerivatives

type
  AdaGradSlow*[L] = ref object of BaseCSROptimizer
    loss: L
    eta0: float64
    eps: float64
    shuffle: bool
    it: int
    g_sum_P: Tensor
    g_norms_P: Tensor
    g_sum_w: Vector
    g_norms_w: Vector
    g_sum_intercept: float64
    g_norms_intercept: float64
    

proc newAdaGradSlow*[L](eta0 = 0.1,  maxIter = 100, alpha0=1e-6, alpha=1e-3,
                        beta=1e-3, loss: L = newSquared(), eps=1e-10,
                        verbose = 1, tol = 1e-3, shuffle = true): AdaGradSlow[L] =
  result = AdaGradSlow[L](eta0: eta0, alpha0: alpha0, alpha: alpha, beta: beta,
                          loss: loss, eps: eps, maxIter: maxIter, tol: tol, it: 1,
                          verbose: verbose, shuffle: shuffle)


proc fit*[L](self: AdaGradSlow[L], X: Matrix, y: seq[float64], fm: var FMSlow) =
  fm.init(X)
  let y = fm.checkTarget(y)
  let
    nSamples = X.shape[0]
    nFeatures = X.shape[1]
    nComponents = fm.P.shape[1]
    nOrders = fm.P.shape[0]
    degree = fm.degree
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    nAugments = fm.nAugments
    loss = self.loss
  var
    indices = toSeq(0..<nSamples)
    dA: Tensor = zeros(fm.P.shape)

  if not fm.warmstart:
    self.it = 1
  if self.it == 1:
    self.g_sum_P = zeros(fm.P.shape)
    self.g_norms_P = zeros(fm.P.shape) + self.eps
    if fm.fitLinear:
      self.g_sum_w = zeros(fm.w.shape)
      self.g_norms_w = zeros(fm.w.shape) + self.eps
    if fm.fitIntercept:
      self.g_sum_intercept = 0.0
      self.g_norms_intercept = self.eps

  for epoch in 0..<self.maxIter:
    if self.shuffle: shuffle(indices)
    for i in indices:
      let it = float(self.it)

      # compute prediction and gradient
      var yPred = fm.intercept
      for j in 0..<nFeatures:
        yPred += fm.w[j] * X[i, j]
      for order in 0..<nOrders:
        yPred += computeAnovaGrad(fm.P, X, i, degree-order, order, 
                                  nAugments, dA)
      let dL = loss.dloss(y[i], yPred)

      # update parameters
      if fitIntercept:
        self.g_sum_intercept += dL
        self.g_norms_intercept += dL^2
        let denom = sqrt(self.g_norms_intercept) + self.eta0 * it * self.alpha0
        fm.intercept = - self.eta0 * self.g_sum_intercept / denom

      if fitLinear:
        let denom = self.eta0 * it * self.alpha 
        for j in 0..<nFeatures:
          self.g_sum_w[j] += dL * X[i, j]
          self.g_norms_w[j] += (dL * X[i, j])^2
          fm.w[j] = - self.eta0 * self.g_sum_w[j]
          fm.w[j] /= (denom + sqrt(self.g_norms_w[j]))
    
      for order in 0..<nOrders:
        let denom = self.eta0 * it * self.beta
        for s in 0..<nComponents:
          for j in 0..<(nFeatures+nAugments):
            let grad = dL*dA[order, s, j]
            self.g_sum_P[order, s, j] += grad 
            self.g_norms_P[order, s, j] += grad*grad
            fm.P[order, s, j] = -self.eta0 * self.g_sum_P[order, s, j]
            fm.P[order, s, j] /= (denom + sqrt(self.g_norms_P[order, s, j]))
      
      inc(self.it)