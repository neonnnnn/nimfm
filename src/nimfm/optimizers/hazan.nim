import ../dataset, ../tensor, ../kernels, ../extmath, ../utils, ../loss
import ../convex_factorization_machine
from ../fm_base import checkTarget, checkInitialized
import optimizer_base
import math, strformat, strutils

type
  Hazan* = ref object of BaseCSCOptimizer
    ## Hazan's algorithm for convex factorization machines with SquaredLoss.
    ## This solver solves not regularized problem but constrained problem
    ## such that trace norm of the interaction weight matrix = eta.
    ## Regularization parameters alpha0, alpha, and beta
    ## in ConvexFactorizationMachine are ignored.
    maxIterPower: int
    tolPower: float64
    optimal: bool
    nTol: int
    it: int # for warm-start with optimal=false


proc newHazan*(
  maxIter = 100, verbose = 2, tol = 1e-7, nTol=10, maxIterPower = 1000,
  tolPower = 1e-7, optimal = true): Hazan =
  ## Creates new Hazan for ConvexFactorizationMachine.
  ## maxIter: Maximum number of iteration for alternative optimization.
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
    maxIter: maxIter, tol: tol, verbose: verbose, nTol: nTol,
    maxIterPower: maxIterPower, tolPower: tolPower, optimal: optimal)


proc fit*(self: Hazan, X: CSCDataset, y: seq[float64],
          cfm: var ConvexFactorizationMachine[Squared]) =
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
    if fitIntercept: 
      w.add(cfm.intercept)
      colNormSq.add(1.0)
      resZ.add(0.0)
    colNormSq += 1e-5

  # compute prediction
  linear(X, cfm.w, yPredLinear)
  yPredLinear += cfm.intercept
  for s in 0..<len(cfm.lams):
    if ignoreDiag: 
      anova(X, cfm.P, cacheK, 2, s, 0)
    else: 
      poly(X, cfm.P, cacheK, 2, s, 0)
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
    mvmul(X, w[0..<nFeatures], Xp)
    if fitIntercept: Xp += w[^1]
    vmmul(Xp, X, result)
    if fitIntercept: result[^1] = sum(Xp)

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
    # fit P
    let (_, p) = powerMethod(linearOpPower, nFeatures, self.maxIterPower,
                             tol=self.tolPower)
    # Add or replace? 
    var s = len(cfm.lams)
    if s == cfm.maxComponents: # replace old p
      s = argmin(cfm.lams) # replace old p whose lambda is minimum
      yPredQuad -= cfm.lams[s] * K[s]
      cfm.P[s] = p
    else: # add new p
      cfm.P.addRow(p)

    if ignoreDiag: 
      anova(X, cfm.P, cacheK, 2, s, 0)
    else:
      poly(X, cfm.P, cacheK, 2, s, 0)
    
    if s != len(cfm.lams): # replace old p
      yPredQuad += cfm.lams[s] * K[s]
      residual = y - yPredQuad - yPredLinear
      K[s] = cacheK[0..^1, 2]
    else:
      K.addRow(cacheK[0..^1, 2]) # add new p
    
    # compute stepsize
    if self.optimal:
      let d = cfm.eta * K[s] - yPredQuad
      stepsize = dot(d, residual) / norm(d, 2)^2
      stepsize = min(max(1e-10, stepsize), 1.0)
    else:
      stepsize = 2.0 / (float(self.it)+2.0)
    # update lams
    if len(cfm.lams) != 0 and sum(cfm.lams) + cfm.eta * stepsize > cfm.eta:
      yPredQuad *= cfm.eta * (1.0 - stepsize) / sum(cfm.lams)
      cfm.lams *= cfm.eta * (1.0 - stepsize) / sum(cfm.lams)
    if s == len(cfm.lams):
      cfm.lams.add(cfm.eta*stepsize)
    else:
      cfm.lams[s] += cfm.eta*stepsize

    # update predictions
    yPredQuad += cfm.eta * stepsize * K[s]

    # fit w
    residual = y - yPredQuad
    if fitLinear:
      # optimize w by conjugate gradient
      resZ[0..<nFeatures] = vmmul(residual, X)
      if fitIntercept:
        resZ[^1] = sum(residual)
      let maggrad = norm(resZ, 1)
      let tolCG = 1e-5 * maggrad
      w *= colNormSq # since we use left and right preconditioning
      cg(linearOpCG, resZ, w, maxIter=1000, preconditioner=preconditioner,
         init=false, tol=tolCG)
      # substitute and update prediction
      cfm.w = w[0..<nFeatures]
      mvmul(X, cfm.w, yPredLinear)
      if fitIntercept:
        cfm.intercept = w[^1]
        yPredLinear += cfm.intercept
    elif fitIntercept:
      cfm.intercept = sum(residual) / float(nSamples)
      yPredLinear[0..<nSamples] = cfm.intercept

    # stopping criterion
    residual = y - yPredQuad - yPredLinear
    lossNew = norm(residual, 2)^2  / float(nSamples)
    if self.verbose > 0:
      let epochAligned = align($(self.it), len($self.maxIter))
      stdout.write(fmt"Epoch: {epochAligned}")
      stdout.write(fmt"   MSE: {lossNew:1.4e}")
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