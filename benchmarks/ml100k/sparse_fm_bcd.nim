import nimfm, utils
import strformat

when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var reg = newSquaredL21()
  let beta = 1e-5
  let gamma = 1e-7

  var fm = newFactorizationMachine(task=regression)
  var cd = newPBCD(
    maxIter=100, shrink=false, maxSearch=0, verbose=1,
    beta=beta, alpha0=1e-10, alpha=1e-6,
    gamma=gamma, reg=reg)
  cd.fit(Xtr, yTr, fm)

  echo("L1 norm of the interaction matrix: ", norm(matmul(fm.P[0].T, fm.P[0]), 1))
  echo("Number of used interactions: ", countInteractions(fm.P[0].T))
  echo("Number of used features: ", countFeatures(fm.P[0].T))
  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))