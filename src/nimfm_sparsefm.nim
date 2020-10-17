# For end users
import nimfm/modules
import strutils, sugar, sequtils, math, tables, strformat


proc echoDataInfo[T](X: BaseDataset[T]) =
  echo("   Number of samples  : ", X.nSamples)
  echo("   Number of features : ", X.nFeatures)
  echo("   Number of non-zeros: ", X.nnz)
  echo("   Maximum value      : ", X.max)
  echo("   Minimum value      : ", X.min)


proc eval(fm: FactorizationMachine, task: TaskKind, test: string,
          predict: string, nFeatures: int, verbose: int) =
  var X: CSRDataset
  var y: seq[float64]
  if verbose > 0: echo("Load test data")
  loadSVMLightFile(test, X, y, nFeatures)
  if verbose > 0: echoDataInfo(X)
  let yPred = fm.decisionFunction(X)
  case task
  of regression:
    echo("Test RMSE: ", rmse(y, yPred))
  of classification:
    echo("Test Accuracy: ",
         accuracy(y.map(x=>sgn(x)), yPred.map(x=>sgn(x))))
  if predict != "":
    var f: File = open(predict, fmWrite)
    for val in fm.decisionFunction(X):
      f.writeLine(val)
    f.close()


proc trainInner2[L, R](fm: var FactorizationMachine, train: string, 
                       alpha0 = 1e-7, alpha = 1e-5, beta = 1e-5, gamma = 1e-5,
                       loss: L=newSquared(), reg: R = newSquaredL12(),
                       solver = "nmapgd", maxIter = 100, tol = 1e-5, eta0 = 0.1,
                       scheduling = optimal, power = 1.0, miniBatchSize = -1,
                       nFeatures = -1, verbose = 1) =
  ## training a factorization machine
  var X: CSRDataset
  var y: seq[float64]
  loadSVMLightFile(train, X, y, nFeatures)
  if verbose > 0: echoDataInfo(X)

  case solver
    of "nmapgd":
      var optim = newNMAPGD(maxIter=maxIter, alpha0=alpha0, alpha=alpha,
                            beta=beta, gamma=gamma, loss=loss, reg=reg,
                            verbose=verbose, tol=tol)
      optim.fit(X, y, fm)
    of "fista":
      var optim = newFISTA(maxIter=maxIter, alpha0=alpha0, alpha=alpha,
                           beta=beta, gamma=gamma, loss=loss, reg=reg,
                           verbose=verbose, tol=tol)
      optim.fit(X, y, fm)
    of "mbpsgd":
      var optim = newMBPSGD(maxIter, eta0, alpha0, alpha, beta, gamma,
                            loss=loss, reg=reg, scheduling=scheduling,
                            power=power, miniBatchSize=miniBatchSize,
                            verbose=verbose, tol=tol)
      optim.fit(X, y, fm)
    of "katyusha":
      var optim = newKatyusha(maxIter, eta0, alpha0, alpha, beta, gamma,
                               loss=loss, reg=reg, miniBatchSize=miniBatchSize,
                               verbose=verbose, tol=tol)
      optim.fit(X, y, fm)
    else:
      raise newException(ValueError, "Solver " & solver & " is not supported")


proc trainPBCD[L, R](fm: var FactorizationMachine, train: string, alpha0 = 1e-7,
                    alpha = 1e-5, beta = 1e-5, gamma = 1e-5,
                    loss: L=newSquared(), reg: R = newSquaredL21(),
                    maxIter = 100, tol = 1e-5, nFeatures = -1, verbose = 1) =
  ## training a factorization machine
  var X: CSCDataset
  var y: seq[float64]
  loadSVMLightFile(train, X, y, nFeatures)
  if verbose > 0: echoDataInfo(X)

  var optim = newPBCD(maxIter=maxIter, alpha0=alpha0, alpha=alpha,
                      beta=beta, gamma=gamma, loss=loss, reg=reg,
                      verbose=verbose, tol=tol)
  optim.fit(X, y, fm)


proc trainPCD[L, R](fm: var FactorizationMachine, train: string, alpha0 = 1e-7,
                    alpha = 1e-5, beta = 1e-5, gamma = 1e-5,
                    loss: L=newSquared(), reg: R = newSquaredL12(),
                    maxIter = 100, tol = 1e-5, nFeatures = -1, verbose = 1) =
  ## training a factorization machine
  var X: CSCDataset
  var y: seq[float64]
  loadSVMLightFile(train, X, y, nFeatures)
  if verbose > 0: echoDataInfo(X)

  var optim = newPCD(maxIter=maxIter, alpha0=alpha0, alpha=alpha,
                     beta=beta, gamma=gamma, loss=loss, reg=reg,
                     verbose=verbose, tol=tol)
  optim.fit(X, y, fm)


proc trainInner[L](fm: var FactorizationMachine, train: string,
                   alpha0 = 1e-7, alpha = 1e-5, beta = 1e-5,
                   gamma = 1e-5, loss: L = newSquared(), reg = "squaredl12",
                   solver = "nmapgd", maxIter = 200, tol = 1e-5, eta0 = 0.1,
                   scheduling = optimal, power = 1.0, miniBatchSize = -1,
                   nFeatures = -1, verbose = 1) =
  ## training a factorization machine
  case reg
    of "squaredl12":
      if solver == "pcd":
        trainPCD(fm, train, alpha0, alpha, beta, gamma, loss, newSquaredL12(),
                 maxIter, tol, nFeatures, verbose)
      elif solver == "pbcd":
        raise newException(ValueError, "PBCD cannot be used for squaredl12.")
      else:
        trainInner2(fm, train, alpha0, alpha, beta, gamma, loss,
                    newSquaredL12(), solver, maxIter, tol, eta0,
                    scheduling, power, miniBatchSize, nFeatures, verbose)
    of "squaredl21":
      if solver == "pcd":
        raise newException(ValueError, "PCD cannot be used for squaredL21.")
      elif solver == "pbcd":
        trainPBCD(fm, train, alpha0, alpha, beta, gamma, loss, newSquaredL21(),
                  maxIter, tol, nFeatures, verbose)
      else:
        trainInner2(fm, train, alpha0, alpha, beta, gamma, loss, newSquaredL21(),
                    solver, maxIter, tol, eta0, scheduling, power, miniBatchSize,
                    nFeatures, verbose)
    of "l1":
      if solver == "pcd":
        trainPCD(fm, train, alpha0, alpha, beta, gamma, loss, newL1(),
                 maxIter, tol, nFeatures, verbose)
      elif solver == "pbcd":
        trainPBCD(fm, train, alpha0, alpha, beta, gamma, loss, newL1(),
                  maxIter, tol, nFeatures, verbose)
      else:
        trainInner2(fm, train, alpha0, alpha, beta, gamma, loss, newL1(),
                    solver, maxIter, tol, eta0, scheduling, power, miniBatchSize,
                    nFeatures, verbose)
    of "l21":
      if solver == "pcd":
        raise newException(ValueError, "PCD cannot be used for L21.")
      elif solver == "pbcd":
        trainPBCD(fm, train, alpha0, alpha, beta, gamma, loss, newL21(),
                  maxIter, tol, nFeatures, verbose)
      else:
        trainInner2(fm, train, alpha0, alpha, beta, gamma, loss, newL21(),
                    solver, maxIter, tol, eta0, scheduling, power, miniBatchSize,
                    nFeatures, verbose)
    else:
      raise newException(ValueError, fmt"regularization {reg} is not supported")


proc train(task: TaskKind, train: string, test = "", degree = 2,
           nComponents = 30, alpha0 = 1e-7, alpha = 1e-5, beta = 1e-5,
           gamma = 1e-5, loss="squared", reg = "squaredl12",
           fitLower = explicit, fitLinear = true,
           fitIntercept = true, scale = 0.1, randomState = 1,
           solver = "mbpsgd", maxIter = 100, tol = 1e-5, eta0 = 0.1,
           scheduling = optimal, power = 1.0, miniBatchSize = -1, threshold=0.1,
           dump = "", load = "", predict = "", nFeatures = -1, verbose = 1) =
  ## training a factorization machine
  var fm: FactorizationMachine
  if load == "":
    fm = newFactorizationMachine(
      task = task, degree = degree, nComponents = nComponents,
      fitLower = fitLower, fitIntercept = fitIntercept, fitLinear = fitLinear,
      warmStart = false, randomState = randomState, scale = scale
    )
  else:
    load(fm, load, true)
  
  case loss
    of "squared":
      trainInner(fm, train, alpha0, alpha, beta, gamma, newSquared(), reg,
                 solver, maxIter, tol, eta0, scheduling,
                 power, miniBatchSize, nFeatures, verbose)
    of "huber":
      trainInner(fm, train, alpha0, alpha, beta, gamma, newHuber(threshold),
                 reg, solver, maxIter, tol, eta0, scheduling,
                 power, miniBatchSize, nFeatures, verbose)
    of "squared_hinge":
      trainInner(fm, train, alpha0, alpha, beta, gamma, newSquaredHinge(),
                 reg, solver, maxIter, tol, eta0, scheduling,
                 power, miniBatchSize, nFeatures, verbose)
    of "logistic":
      trainInner(fm, train, alpha0, alpha, beta, gamma, newLogistic(),
                reg, solver, maxIter, tol, eta0, scheduling,
                 power, miniBatchSize, nFeatures, verbose)
    else:
      raise newException(ValueError, fmt"loss {loss} is not supported")


  if test != "": eval(fm, task, test, predict, nFeatures, verbose)

  if dump != "": fm.dump(dump)


proc testInner[L](task: TaskKind, test, load: string, dump = "", 
                  loss: L = newSquared(), predict = "", nFeatures = -1,
                  verbose = 1) =
  var fm: FactorizationMachine
  load(fm, load, false)
  eval(fm, task, test, predict, nFeatures, verbose)
  if dump != "": fm.dump(dump)


proc test(task: TaskKind, test, load: string, dump = "",  loss="squared", 
          predict = "", nFeatures = -1, verbose = 1) =
  ## test a factorization machine
  case loss
    of "squared":
      testInner(task, test, load, dump, newSquared(), predict, nFeatures, verbose)
    of "huber":
      testInner(task, test, load, dump, newHuber(), predict, nFeatures, verbose)
    of "squared_hinge":
      testInner(task, test, load, dump, newSquaredHinge(), predict, nFeatures, 
                verbose)
    of "logistic":
      testInner(task, test, load, dump, newLogistic(), predict, nFeatures, verbose)
    else:
      raise newException(ValueError, fmt"loss {loss} is not supported.")


when isMainModule:
  import cligen; include cligen/mergeCfgEnv

  const hlUse = "\e[7m$command $args\e[0m\n\e[1m${doc}\e[0mOptions:\n$options"
  {.push hint[GlobalVar]: off.}
  const nimbleFile = staticRead "../nimfm.nimble"
  clCfg.version = nimbleFile.fromNimble("version")
  var docLine = "nimfm for sparse FMs.\n\n"
  let topLvlUse = """${doc}Usage:
  $command {SUBCMD}  [sub-command options & parameters]

SUBCMDs:
$subcmds
$command {-h|--help} or with no args at all prints this message.
Run "$command {help SUBCMD|SUBCMD --help}" to see help for just SUBCMD.
Run "$command help" to get *comprehensive* help.$ifVersion"""
  clCfg.reqSep = true

  var noVsn = clCfg
  {.pop.}
  noVsn.version = ""
  dispatchMulti(
    ["multi", doc = docLine, usage = topLvlUse],
    [train, help = {
     "task": "r for regression and c for binary classification",
     "train": "Filename of training data (libsvm/svmlight format)",
     "test": "Filename of test data (libsvm/svmlight format)",
     "degree": "Degree of the polynomial (order of feature interactions)",
     "n-components": "Number of basis vectors (rank hyperparameter)",
     "alpha0": "Regularization-strength for intercept (bias) term",
     "alpha": "Regularization sterngth for linear term",
     "beta": "Regularization-strength for interaction term",
     "gamma": "Sparsity-inducing-regularization-strength for interaction term",
     "loss": "Optimized loss function, squared, huber, squared_hinge, " &
             "logistic, or huber",
     "reg": "Sparsity-inducing regularization, squaredl12, squaredl21, l1 or l21",
     "fit-lower": "Whether and how to fit lower-order terms. " &
                  "explicit, augment, or none",
     "fit-linear": "Whether to fit linear term or not (0 or 1)",
     "fit-intercept": "Whether to fit intercept term or not (0 or 1)",
     "scale": "Standard derivation for initialization of interaction weights",
     "random-state": "Seed od the pseudo random number generator. " &
                     "0 is not allowed",
     "solver": "Optimization method, pcd, pbcd, nmapgd, fista, mbpsgd, " & 
               "or katyusha",
     "maxIter": "Maximum number of optimization iteration (epoch)",
     "tol": "Tolerance for stopping criterion",
     "eta0": "Step-size parameter for sgd",
     "scheduling": "Step-size scheduling method for sgd. " &
                   "optimal, constant, pegasos, or invscaling",
     "power": "Hyperparameter for step-size scheduling",
     "mini-batch-size": "Number of samples in one mini-batch. If <= 0, then " &
                        "mini-batch-size is automatically determined",
     "threshold": "Hyperparameter for huber loss",
     "dump": "Filename for dumping model",
     "load": "Filename for loading model",
     "predict": "Filename for prediction of test data",
     "n-features": "Number of features. If -1 (default), the maxium index " &
                   "in training data is used",
     "verbose": "Verbosity level (0 or 1)"
    },
    usage = hluse,
    short = {"": '\0'}],
    [test, help = {
     "task": "r for regression and c for binary classification",
     "test": "Filename of test data",
     "load": "Filename for loading model",
     "dump": "Filename for dumping model",
     "predict": "Filename for prediction of test data",
     "n-features": "Number of features. If -1 (default), the maxium index " &
                   "in training data is used",
     "verbose": "Whether to print information"
    }, usage = hluse, short = {"": '\0'}])