import ../dataset, ../tensor/tensor, ../kernels, ../extmath, ../utils
import ../models/convex_factorization_machine
from ../models/fm_base import checkTarget, checkInitialized
import optimizer_base
import math, strformat, strutils, sugar

type
  Hazan* = ref object of BaseCSCOptimizer
    ## Hazan's algorithm for convex factorization machines with SquaredLoss.
    ## This solver solves not regularized problem but constrained problem
    ## such that trace norm of the interaction weight matrix = eta.
    ## Regularization parameters alpha0, alpha, and beta
    ## in ConvexFactorizationMachine are ignored.
    eta: float64
    maxIterPower: int
    tolPower: float64
    optimal: bool
    nTol: int
    it: int # for warm-start with optimal=false


proc newHazan*(
  maxIter = 100, eta=1000.0, verbose = 2, tol = 1e-7,
  nTol=10, maxIterPower = 1000, tolPower = 1e-7, optimal = true): Hazan =
  ## Creates new Hazan for ConvexFactorizationMachine.
  ## maxIter: Maximum number of iteration for alternative optimization.
  ## eta: Regularization-constraint for the trace norm of
  ##      the feature interaction matrix.
  ## maxIterPower: Maximum number of iteration for power iteration.
  ## verbose: Whether to print information on optimization processes.
  ## tol: Tolerance hyperparameter for stopping criterion.
  ##      If the decreasing of the objective value is smaller than tol
  ##      with nTol times in a row, then this solver stops.
  ## nTol: Tolerance hyperparameter for stopping criterion.
  ##       If the decreasing of the objective value is smaller than tol
  ##       with nTol times in a row, then this solver stops.
  ## tolPower: Tolerance hyperparameter for stopping criterion 
  ##           of power iteration.
  ## optimal: Whether to use optimal step-size or not (2.0 / (t+2.0)).
  ##          If optimal is true, this solver does not stop optimization
  ##          even when nComponents > maxComponents and replaces 
  ##          the old basis vector whose lam is minimum with the new one.
  result = Hazan(
    maxIter: maxIter, eta:eta, tol: tol,
    verbose: verbose, nTol: nTol, maxIterPower: maxIterPower,
    tolPower: tolPower, optimal: optimal)


proc computeStepSize(self: Hazan, K: Matrix, yPredQuad, residual: Vector,
                     s: int): float64 =
  if self.optimal:
    let d = self.eta * K[s] - yPredQuad
    result = dot(d, residual) / norm(d, 2)^2
    result = min(max(1e-10, result), 1.0)
  else:
    result = 2.0 / (float(self.it)+2.0)


proc fit*(self: Hazan, X: ColDataset, y: seq[float64],
          cfm: ConvexFactorizationMachine,
          callback: (Hazan, ConvexFactorizationMachine)->void = nil) =
  ## Fits the factorization machine on X and y by coordinate descent.
  cfm.init(X)
  let y = checkTarget(cfm, y)
  let
    nSamples = X.nSamples
    nFeatures = X.nFeatures
    fitLinear = cfm.fitLinear
    fitIntercept = cfm.fitIntercept
    ignoreDiag = cfm.ignoreDiag

  # caches
  var
    yPredLinear: Vector = zeros([nSamples])
    yPredQuad: Vector = zeros([nSamples])
    residual: Vector = zeros([nSamples])
    cacheK: Matrix = zeros([nSamples, 3])
    K: Matrix = zeros([len(cfm.lams), nSaMples])
    w: Vector
    Xp = zeros([nSamples])
    resZ: Vector
    colNormSq: Vector
    isConverged = false
    lossOld = 0.0
    lossNew = 0.0

  if not cfm.warmStart:
    self.it = 0

  if fitLinear:
    w = cfm.w[0..<nFeatures]
    resZ = zeros([nFeatures])
    colNormSq = norm(X, p=2, axis=0)
    colNormSq *= colNormSq
    if fitIntercept: 
      w.add(cfm.intercept)
      colNormSq.add(float(nSamples))
      resZ.add(0.0)
    colNormSq += 1e-5

  # compute prediction
  linear(X, cfm.w, yPredLinear)
  yPredLinear += cfm.intercept
  for s in 0..<len(cfm.lams):
    if ignoreDiag: 
      anova(X, cfm.P, cacheK, 2, s)
    else: 
      poly(X, cfm.P, cacheK, 2, s)
    K[s] = cacheK[0..^1, 2]
    yPredQuad += cfm.lams[s] * K[s]


  # linear map for PowerMethod
  proc linearOpPower(p: Vector, result: var Vector) =
    mvmul(X, p, Xp)
    Xp *= residual
    vmmul(Xp, X, result)
    if ignoreDiag:
      for j in 0..<nFeatures:
        for (i, val) in X.getCol(j):
          result[j] -= val*val*residual[i]*p[j]

  # linear map for CG
  proc linearOpCG(w: Vector, result: var Vector) =
    mvmul(X, w, Xp)
    vmmul(Xp, X, result)

  # preconditioner for CG
  proc preconditioner(w: var Vector) = w /= colNormSq

  # start optimization
  residual = y - yPredQuad - yPredLinear
  lossOld = norm(residual, 2)^2 / float(nSamples)
  var stepsize = 0.0
  var nTol = 0
  for it in 0..<self.maxIter:
    if not self.optimal and len(cfm.lams) >= cfm.maxComponents:
      break

    #### fit P ####
    let (_, p) = powerMethod(linearOpPower, nFeatures, self.maxIterPower,
                             tol=self.tolPower)
    
    # Add or replace? 
    var s = len(cfm.lams)
    if s == cfm.maxComponents: # replace old p
      s = argmin(cfm.lams)
      cfm.P[s] = p
      yPredQuad -= cfm.lams[s] * K[s]
    else: # add new p
      cfm.P.addRow(p)
      cfm.lams.add(0.0)
      K.addRow(zeros([nFeatures]))

    # update K, yPredQuad, and residual
    if ignoreDiag: 
      anova(X, cfm.P, cacheK, 2, s)
    else:
      poly(X, cfm.P, cacheK, 2, s)
    K[s] = cacheK[0..^1, 2]
    yPredQuad += cfm.lams[s] * K[s]
    residual = y - yPredQuad - yPredLinear

    # compute stepsize and update lams/yPredQuad
    stepsize = computeStepSize(self, K, yPredQuad, residual, s)
    cfm.lams *= (1 - stepsize)
    yPredQuad *= (1 - stepsize)
    cfm.lams[s] += self.eta * stepsize
    yPredQuad += self.eta * stepsize * K[s]

    # re-scale
    if sum(cfm.lams) > self.eta:
      yPredQuad *= self.eta  / sum(cfm.lams)
      cfm.lams *= self.eta  / sum(cfm.lams)

    #### fit linear and intercept ####
    residual = y - yPredQuad
    if fitLinear:
      if fitIntercept:
        X.addDummyFeature(1.0, 1)
      # optimize w by conjugate gradient
      vmmul(residual, X, resZ)
      let maggrad = norm(resZ, 1)
      let tolCG = 1e-5 * maggrad
      w *= colNormSq # since we use left and right preconditioning
      cg(linearOpCG, resZ, w, maxIter=1000, preconditioner=preconditioner,
         init=false, tol=tolCG)
      # substitute and update prediction
      cfm.w = w[0..<nFeatures]
      mvmul(X, w, yPredLinear)
      if fitIntercept:
        X.removeDummyFeature(1)
        cfm.intercept = w[^1]
    elif fitIntercept:
      cfm.intercept = sum(residual) / float(nSamples)
      yPredLinear[0..<nSamples] = cfm.intercept
    
    if not callback.isNil:
      callback(self, cfm)
    # stopping criterion
    residual = y - yPredQuad - yPredLinear
    lossNew = norm(residual, 2)^2  / float(nSamples)
    if self.verbose > 0:
      let epochAligned = align($(self.it), len($self.maxIter))
      stdout.write(fmt"Epoch: {epochAligned}")
      stdout.write(fmt"   MSE/2: {lossNew/2.0:1.4e}")
      stdout.write(fmt"   Trace Norm: {norm(cfm.lams, 1):1.4e}")
      stdout.write("\n")
    
    if lossOld - lossNew < self.tol:
      inc(nTol)
      if ntol >= self.nTol:
        if self.verbose > 0:
          echo("Converged at iteration ", self.it+1, ".")
        isConverged = true
        break
    else:
      nTol = 0
  
    lossOld = lossNew
    inc(self.it)

  if not isConverged and self.verbose > 0:
    echo("Objective did not converge. Increase maxIter.")