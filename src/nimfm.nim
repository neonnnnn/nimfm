import nimfm/factorization_machine
import nimfm/convex_factorization_machine
import nimfm/fm_base
import nimfm/optimizers/coordinate_descent
import nimfm/optimizers/sgd
import nimfm/optimizers/greedy_coordinate_descent
import nimfm/loss
import nimfm/dataset
import nimfm/metrics
import nimfm/tensor

import strutils, sugar, sequtils, math

export fm_base, factorization_machine, loss, dataset, metrics, tensor
export sgd, coordinate_descent
export convex_factorization_machine, greedy_coordinate_descent

# The followings are for end users
proc echoDataInfo(X: BaseDataset) =
  echo("   Number of samples  : ", X.nSamples)
  echo("   Number of features : ", X.nFeatures)
  echo("   Number of non-zeros: ", X.nnz)
  echo("   Maximum value      : ", X.max)
  echo("   Minimum value      : ", X.min)


proc eval(fm: FactorizationMachine, task: TaskKind, test: string,
          predict: string, nFeatures: int, verbose: int) =
  var X: CSRDataset
  var y: seq[float64]
  if verbose > 0: echo("Load test data.")
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


proc train(task: TaskKind, train: string, test = "", degree = 2,
           nComponents = 30, alpha0 = 1e-7, alpha = 1e-5, beta = 1e-5,
           loss = Squared, fitLower = explicit, fitLinear = true,
           fitIntercept = true, scale = 0.1, randomState = 1,
           solver = "cd", maxIter = 100, tol = 1e-5, eta0 = 0.1,
           scheduling = optimal, powerIt = 1.0, dump = "", load = "",
           predict = "", nFeatures = -1, verbose = 1) =
  ## training a factorization machine
  var fm: FactorizationMachine
  if load == "":
    fm = newFactorizationMachine(
      task = task, degree = degree, nComponents = nComponents, alpha0 = alpha0,
      alpha = alpha, beta = beta, loss = loss, fitLower = fitLower,
      fitIntercept = fitIntercept, fitLinear = fitLinear, warmStart = false,
      randomState = randomState, scale = scale
    )
  else:
    load(fm, load, true)
  case solver
    of "cd", "als":
      var X: CSCDataset
      var y: seq[float64]
      loadSVMLightFile(train, X, y, nFeatures)
      if verbose > 0: echoDataInfo(X)
      var optim = newCoordinateDescent(maxIter, verbose, tol)
      optim.fit(X, y, fm)
    of "sgd":
      var X: CSRDataset
      var y: seq[float64]
      loadSVMLightFile(train, X, y, nFeatures)
      if verbose > 0: echoDataInfo(X)
      var optim = newSGD(eta0, scheduling, powerIt, maxIter, verbose, tol)
      optim.fit(X, y, fm)
    else:
      raise newException(ValueError, "Solver " & solver & " is not supported.")

  if test != "": eval(fm, task, test, predict, nFeatures, verbose)

  if dump != "": fm.dump(dump)


proc test(task: TaskKind, test, load: string, dump = "",
          predict = "", nFeatures = -1, verbose = 1) =
  ## test a factorization machine
  var fm: FactorizationMachine
  load(fm, load, false)
  eval(fm, task, test, predict, nFeatures, verbose)
  if dump != "": fm.dump(dump)


when isMainModule:
  import cligen; include cligen/mergeCfgEnv

  const hlUse = "\e[7m$command $args\e[0m\n\e[1m${doc}\e[0mOptions:\n$options"
  {.push hint[GlobalVar]: off.}
  const nimbleFile = staticRead "../nimfm.nimble"
  clCfg.version = nimbleFile.fromNimble("version")
  let docLine = nimbleFile.fromNimble("description") & "\n\n"
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
     "n-components": "Number of basis vectors (rank hyper-parameter)",
     "alpha0": "Regularization strength for intercept (bias) term",
     "alpha": "Regularization sterngth for linear term",
     "beta": "Regularization strength for interaction term",
     "loss": "Optimized loss function, Squared, SquaredHinge, or Logistic. " &
             "Ignored when --task:r",
     "fit-lower": "Whether and how to fit lower-order terms. " &
                  "explicit, augment, or none.",
     "fit-linear": "Whether to fit linear term or not (0 or 1)",
     "fit-intercept": "Whether to fit intercept term or not (0 or 1)",
     "scale": "Standard derivation for initialization of interaction weights",
     "random-state": "Seed od the pseudo random number generator. " &
                     "0 is not allowed.",
     "solver": "Optimization method, cd, als (they are same), or sgd",
     "maxIter": "Maximum number of optimization iteration (epoch)",
     "tol": "Tolerance for stopping criterion",
     "eta0": "Step-size parameter for sgd",
     "scheduling": "Step-size scheduling method for sgd. " &
                   "optimal, constant, pegasos, or invscaling",
     "power-it": "Hyper-parameter for step-size scheduling",
     "dump": "Filename for dumping model",
     "load": "Filename for loading model",
     "predict": "Filename for prediction of test data",
     "n-features": "Number of features. If -1 (default), the maxium index " &
                   "in training data is used.",
     "verbose": "Verbosity level (0 or 1)."
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
                   "in training data is used.",
     "verbose": "Whether to print information"
    }, usage = hluse, short = {"": '\0'}])
