import nimfm/tensor/tensor, nimfm/loss
import nimfm/optimizer/optimizer_base
import sequtils, math, random
import ../model/fm_slow

type
  AdaGradSlow*[L] = ref object of BaseCSROptimizer
    loss*: L
    eta0: float64
    eps: float64
    shuffle*: bool
    it*: int
    g_sum_P*: Tensor
    g_norms_P*: Tensor
    g_sum_w*: Vector
    g_norms_w*: Vector
    g_sum_intercept*: float64
    g_norms_intercept*: float64
    

proc newAdaGradSlow*[L](eta0 = 0.1,  maxIter = 100, alpha0=1e-6, alpha=1e-3,
                        beta=1e-3, loss: L = newSquared(), eps=1e-10,
                        verbose = 1, tol = 1e-3, shuffle = true): AdaGradSlow[L] =
  result = AdaGradSlow[L](eta0: eta0, alpha0: alpha0, alpha: alpha, beta: beta,
                          loss: loss, eps: eps, maxIter: maxIter, tol: tol, it: 1,
                          verbose: verbose, shuffle: shuffle)


proc init*[L](self: AdaGradSlow[L], P: Tensor, w: Vector,
              warmStart, fitLinear, fitIntercept: bool) =
  if not warmstart:
    self.it = 1
  if self.it == 1:
    self.g_sum_P = zeros(P.shape)
    self.g_norms_P = zeros(P.shape) + self.eps
    if fitLinear:
      self.g_sum_w = zeros(w.shape)
      self.g_norms_w = zeros(w.shape) + self.eps
    if fitIntercept:
      self.g_sum_intercept = 0.0
      self.g_norms_intercept = self.eps
  

proc update*[L](self: AdaGradSlow[L], X: Matrix, i: int, P: var Tensor, w: var Vector,
                intercept: var float64, grad: Tensor, dL: float64,
                fitLinear, fitIntercept: bool) =
  let it = float(self.it)

  if fitIntercept:
    self.g_sum_intercept += dL
    self.g_norms_intercept += dL^2
    let denom = sqrt(self.g_norms_intercept) + self.eta0 * it * self.alpha0
    intercept = - self.eta0 * self.g_sum_intercept / denom

  if fitLinear:
    let denom = self.eta0 * it * self.alpha 
    for j in 0..<X.shape[1]:
      self.g_sum_w[j] += dL * X[i, j]
      self.g_norms_w[j] += (dL * X[i, j])^2
      w[j] = - self.eta0 * self.g_sum_w[j]
      w[j] /= (denom + sqrt(self.g_norms_w[j]))
    
  for order in 0..<P.shape[0]:
    let denom = self.eta0 * it * self.beta
    for s in 0..<P.shape[1]:
      for j in 0..<P.shape[2]:
        self.g_sum_P[order, s, j] += grad[order, s, j]
        self.g_norms_P[order, s, j] += grad[order, s, j]^2
        P[order, s, j] = -self.eta0 * self.g_sum_P[order, s, j]
        P[order, s, j] /= (denom + sqrt(self.g_norms_P[order, s, j]))


proc fit*[L](self: AdaGradSlow[L], X: Matrix, y: seq[float64], fm: var FMSlow) =
  fm.init(X)
  let y = fm.checkTarget(y)
  let
    nSamples = X.shape[0]
    fitLinear = fm.fitLinear
    fitIntercept = fm.fitIntercept
    loss = self.loss
  var
    indices = toSeq(0..<nSamples)
    grad: Tensor = zeros(fm.P.shape)

  init(self, fm.P, fm.w, fm.warmStart, fm.fitLinear, fm.fitIntercept)

  for epoch in 0..<self.maxIter:
    if self.shuffle: shuffle(indices)
    for i in indices:
      # compute prediction
      let yPred = decisionFunction(fm, X, i)
      
      # compute gradient
      let dL = loss.dloss(y[i], yPred)
      grad <- 0.0
      computeGrad(fm, X, i, dL, grad)

      # update parameters
      update(self, X, i, fm.P, fm.w, fm.intercept, grad, dL,
             fitLinear, fitIntercept)
      
      inc(self.it)