import ../dataset, ../tensor/tensor, ../loss
import ../model/factorization_machine, ../model/fm_base
import fit_linear, optimizer_base, utils
from sgd import getEta, SchedulingKind, predictWithGrad
import ../loss
from ../regularizer import newSquaredL12
import sequtils, math, random, sugar


type
  PSGD*[L, R] = ref object of BaseCSROptimizer
    gamma*: float64
    loss*: L
    reg*: R
    eta0: float64
    scheduling: SchedulingKind
    power: float64
    it: int
    shuffle: bool
    nCalls: int


proc newPSGD*[L, R](maxIter=100, eta0 = 0.01, alpha0=1e-6, alpha=1e-3,
                    beta=1e-4, gamma=1e-4, loss: L=newSquared(),
                    reg: R=newSquaredL12(), scheduling = optimal, power = 1.0,
                    verbose = 1, tol = 1e-3, shuffle = true, 
                    nCalls = -1): PSGD[L, R] =
  ## Creates new PSGD.
  ## maxIter: Maximum number of iteration. At each iteration, 
  ##          all parameters are updated nSamples times.
  ## eta0: Step-size parameter.
  ## alpha0: Regularization-strength for intercept.
  ## alpha: Regularization-strength for linear term.
  ## beta: Regularization-strength for higher-order weights.
  ## gamma: Sparsity-inducing-regularization-strength for higher-order weights.
  ## loss: Loss function. It must have mu: float64 field and
  ##       loss/dloss proc: (float64, float64)->float64.
  ## reg: Sparsity-inducing regularization.
  ## scheduling: How to change the step-size.
  ##  - constant: eta = eta0,
  ##  - optimal: eta = eta0 / pow(1+eta0*regul*it, power),
  ##  - invscaling: eta = eta0 / pow(it, power),
  ##  - pegasos: eta = 1.0 / (regul * it),
  ##  where regul is the Regularization-strength hyperparameter.
  ## power: Hyperparameter for step size scheduling.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  ## shuffle: How to choose one instance: cyclic (false) or random permutation
  ##          (true).
  ## nCalls: Frequency with which callback must be called in the inner loop.
  ##         If nCalls <= 0, callback is called per one epoch.
  result = PSGD[L, R](
    maxIter: maxIter, alpha0: alpha0, alpha: alpha, beta: beta, gamma: gamma,
    loss: loss, reg: reg, eta0: eta0, scheduling: scheduling, power: power,
    it: 1, tol: tol, verbose: verbose, shuffle: shuffle, nCalls: nCalls)


proc finalize[L, R](self: PSGD[L, R], sfm: FactorizationMachine, P: var Tensor,
                    scaling_w: float64, scalings_w: var Vector) =
  let
    nOrders = P.shape[0]

  if sfm.fitLinear:
    sfm.w *= scaling_w
    sfm.w /= scalings_w
    scalings_w[0..^1] = scaling_w

  self.reg.lazyUpdateFinal(P, self.beta, self.gamma, sfm.degree)

  for order in 0..<nOrders:
    for j in 0..<P.shape[1]:
      for s in 0..<P.shape[2]:
        sfm.P[order, s, j] = P[order, j, s]


proc fit*[L, R](self: PSGD[L, R], X: RowDataset, y: seq[float64],
                sfm: FactorizationMachine,
                callback: (PSGD[L, R], FactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by stochastic gradient descent.
  sfm.init(X)
  let y = sfm.checkTarget(y)
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    nComponents = sfm.P.shape[1]
    nOrders = sfm.P.shape[0]
    degree = sfm.degree
    alpha0 = self.alpha0
    alpha = self.alpha
    beta = self.beta
    gamma = self.gamma
    reg = self.reg
    fitLinear = sfm.fitLinear
    fitIntercept = sfm.fitIntercept
    nAugments = sfm.nAugments
  var
    scaling_w = 1.0
    scalings_w = newSeqWith(nFeatures, 1.0)
    A: Matrix = zeros([nComponents, degree+1])
    indices = toSeq(0..<nSamples)
    P: Tensor = zeros([nOrders, sfm.P.shape[2], nComponents])
    dA: Tensor = zeros(P.shape)
    isConverged = false
    runningLossOld = 0.0
  
  if not sfm.warmstart:
    self.it = 1
  # copy for fast training
  for order in 0..<nOrders:
    P[order] = sfm.P[order].T
  
  reg.initSGD(degree, nFeatures+nAugments, nComponents)
  
  if self.verbose > 0: # echo header
    echoHeader(self.maxIter, viol=false)

  for epoch in 0..<self.maxIter:
    var runningLoss = 0.0
    var regVal = 0.0
    if X.nCached == X.nSamples and self.shuffle: shuffle(indices)

    for i in indices:
      # synchronize (lazily update) and compute prediction
      for (j, val) in X.getRow(i):
        sfm.w[j] *= scaling_w / scalings_w[j]
      X.addDummyFeature(1.0, nAugments)
      for order in 0..<nOrders:
        reg.lazyUpdate(P[order], beta, gamma, degree, X, i)
      X.removeDummyFeature(nAugments)
      let yPred = predictWithGrad(X, i, P, sfm.w, sfm.intercept, A, dA,
                                  degree, nAugments)
      runningLoss += self.loss.loss(y[i], yPred)

      # update parameters and caches for lazily updates
      let eta_w = self.getEta(alpha)
      let eta_P = self.getEta(beta)
      let eta_P_scaled = eta_P / (1.0 + eta_P * beta)
      let dL = self.loss.dloss(y[i], yPred)
      
      # for P
      X.addDummyFeature(1.0, nAugments)
      for order in 0..<nOrders:
        let indices = X.getRowIndices(i)
        reg.step(P[order], dA[order], dL, beta, gamma, eta_P_scaled, 
                 degree-order, indices)
      reg.updateCacheSGD(eta_P, beta, gamma, degree, X, i)
      X.removeDummyFeature(nAugments)

      # for w and intercept
      if fitIntercept:
        let update = self.getEta(alpha0) * (dL + alpha0 * sfm.intercept)
        sfm.intercept -= update / (1.0 + self.getEta(alpha0)*alpha0)
      if fitLinear:
        discard fitLinearSGD(sfm.w, X, i, alpha, dL, eta_w/(1.0+eta_w*alpha))

      # cache updating is differ from sgd.nim (scaling_w *= (1-eta_w*alpha))
      scaling_w /= (1.0+eta_w*alpha)
      for (j, _) in X.getRow(i):
        scalings_w[j] = scaling_w
      
      # reset scalings in order to avoid numerical error
      if fitLinear and scaling_w < 1e-9:
        sfm.w *= scaling_w
        sfm.w /= scalings_w
        scalings_w[0..^1] = 1.0
        scaling_w = 1.0
      reg.resetCacheSGD(P, gamma, degree)

      # reset dA for next iteration
      for order in 0..<nOrders:
        for (j, _) in X.getRow(i):
          for s in 0..<nComponents:
            dA[order, j, s] = 0.0
        for j in 0..<nAugments: # for dummy feature
          for s in 0..<nComponents:
            dA[order, j, s] = 0.0
      
      # callback
      if self.nCalls > 0 and self.it mod self.nCalls == 0:
        if not callback.isNil:
          finalize(self, sfm, P, scaling_w, scalings_w)
          callback(self, sfm)
      
      inc(self.it)

    runningLoss /= float(nSamples)
    
    # one epoch done
    # callback
    if not callback.isNil and self.nCalls <= 0:
      finalize(self, sfm, P, scaling_w, scalings_w)
      callback(self, sfm)
    
    if runningLoss.classify == fcNan:
      echo("Loss is NaN. Use smaller learning rate.")
      break
    
    if abs(runningLoss - runningLossOld) < self.tol:
      if self.verbose > 0: echo("Converged at epoch ", epoch, ".")
      isConverged = true
      break

    if self.verbose > 0:
      regVal = regularization(P, sfm.w, sfm.intercept, alpha0, alpha, beta)
      for order in 0..<nOrders:
        regVal += gamma * self.reg.eval(P[order], sfm.degree-order)
      echoInfo(epoch+1, self.maxIter, -1, runningLoss, regVal)

    runningLossOld = runningLoss

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")

  # finalize
  finalize(self, sfm, P, scaling_w, scalings_w)
