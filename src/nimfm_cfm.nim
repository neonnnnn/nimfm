import nimfm/modules
import strutils, sugar, sequtils, math, tables, strformat


# The followings are for end users
proc echoDataInfo[T](X: BaseDataset[T]) =
  echo("   Number of samples  : ", X.nSamples)
  echo("   Number of features : ", X.nFeatures)
  echo("   Number of non-zeros: ", X.nnz)
  echo("   Maximum value      : ", X.max)
  echo("   Minimum value      : ", X.min)


proc eval(fm: ConvexFactorizationMachine, task: TaskKind, test: string,
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


proc trainInner[L](cfm: var ConvexFactorizationMachine, train: string,
                   alpha0 = 1e-7, alpha = 1e-5, beta = 1e-5, eta=1000.0,
                   loss: L=newSquared(), solver = "gcd", maxIter = 100,
                   tol = 1e-5, maxIterPower=100, maxIterInner=10,
                   nRefitting=10, refitFully=false, sigma=1e-4,
                   maxIterADMM=100, optimal=true, nFeatures = -1,
                   verbose = 1) =
  ## training a factorization machine
  var X: CSCDataset
  var y: seq[float64]
  loadSVMLightFile(train, X, y, nFeatures)
  if verbose > 0: echoDataInfo(X)

  case solver
    of "gcd":
      var optim: GreedyCD[L] = newGreedyCD(
        maxIter=maxIter, alpha0=alpha0, alpha=alpha, beta=beta, loss=loss,
        maxIterPower=maxIterPower, nRefitting=nRefitting, refitFully=refitFully,
        sigma=sigma, maxIterADMM=maxIterADMM, verbose=verbose, tol=tol)
      optim.fit(X, y, cfm)
    of "hazan":
      if verbose > 0: echoDataInfo(X)
      var optim = newHazan(maxIter, eta=eta, maxIterPower=maxIterPower,
                           optimal=optimal, verbose=verbose, tol=tol)
      optim.fit(X, y, cfm)
    else:
      raise newException(ValueError, "Solver " & solver & " is not supported")


proc train(task: TaskKind, train: string, test = "",
           maxComponents = 30, alpha0 = 1e-7, alpha = 1e-5, beta = 1e-5,
           eta=1000.0, loss="squared",fitLinear = true,
           fitIntercept = true, ignoreDiag=false, solver = "gcd",
           maxIter = 100, tol = 1e-5,
           maxIterPower=100, maxIterInner=10, nRefitting=10, 
           refitFully=false, sigma=1e-4, maxIterADMM=100, optimal=true, 
           threshold=0.1, dump = "", load = "", predict = "",
           nFeatures = -1, verbose = 1) =
  ## training a convex factorization machine
  var cfm: ConvexFactorizationMachine
  if load == "":
    cfm = newConvexFactorizationMachine(
      task = task, maxComponents = maxComponents, ignoreDiag = ignoreDiag,
      fitIntercept = fitIntercept, fitLinear = fitLinear,
      warmStart = false
    )
  else:
    load(cfm, load, true)
  case loss
    of "squared":
      trainInner(cfm, train, alpha0, alpha, beta, eta,
                newSquared(), solver, maxIter, tol, maxIterPower, maxIterInner, nRefitting,
                refitFully, sigma, maxIterADMM, optimal, nFeatures, verbose)
    of "huber":
      trainInner(cfm, train, alpha0, alpha, beta, eta,
                newHuber(threshold), solver, maxIter, tol, maxIterPower, maxIterInner, nRefitting,
                refitFully, sigma, maxIterADMM, optimal, nFeatures, verbose)
    of "squared_hinge":
      trainInner(cfm, train, alpha0, alpha, beta, eta,
                newSquaredHinge(), solver, maxIter, tol, maxIterPower, maxIterInner, nRefitting,
                refitFully, sigma, maxIterADMM, optimal, nFeatures, verbose)
    of "logistic":
      trainInner(cfm, train, alpha0, alpha, beta, eta, newLogistic(), solver,
                 maxIter, tol, maxIterPower, maxIterInner,
                 nRefitting, refitFully, sigma, maxIterADMM, optimal,
                 nFeatures, verbose)
    else:
      raise newException(ValueError, fmt"loss {loss} is not supported")
  
  if test != "": eval(cfm, task, test, predict, nFeatures, verbose)

  if dump != "": cfm.dump(dump)


proc testInner[L](task: TaskKind, test, load: string, dump = "", 
                  loss: L = newSquared(), predict = "", nFeatures = -1,
                  verbose = 1) =
  var fm: ConvexFactorizationMachine
  load(fm, load, false)
  eval(fm, task, test, predict, nFeatures, verbose)
  if dump != "": fm.dump(dump)


proc test(task: TaskKind, test, load: string, dump = "",  loss="squared", 
          predict = "", nFeatures = -1, verbose = 1) =
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
  let docLine = "nimfm for convex FMs.\n\n"
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
     "max-components": "Maximum number of basis vectors (rank hyperparameter)",
     "alpha0": "Regularization-strength for intercept (bias) term",
     "alpha": "Regularization sterngth for linear term",
     "beta": "Regularization-strength for interaction term (gcd)",
     "eta": "Regularization-constraint for interaction term (hazan)",
     "loss": "Optimized loss function, squared, huber, squared_hinge, " &
             "logistic, or huber",
     "fit-linear": "Whether to fit linear term or not (0 or 1)",
     "fit-intercept": "Whether to fit intercept term or not (0 or 1)",
     "ignoreDiag": "Whether to use interactions from same features (e.g, x1^2) or not",
     "solver": "Optimization method, gcd or hazan",
     "maxIter": "Maximum number of optimization iteration (epoch)",
     "tol": "Tolerance for stopping criterion",
     "maxIterPower": "Maximum number of iteration in power method",
     "maxIterInner": "Maximum number of innner iteration (gcd)",
     "nRefitting": "Frequency of the refitting lams and P in inner loop (gcd)",
     "refitFully": "Whether to refit both P and lams or only lams (0 or 1)",
     "sigma": "Parameter for line search (gcd and refitFully=1)",
     "maxIterADMM": "Maximum number of ADMM iteration (gcd and refitFully=1)",
     "optimal": "Whether to use optimal step size or not (hazan)",
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
