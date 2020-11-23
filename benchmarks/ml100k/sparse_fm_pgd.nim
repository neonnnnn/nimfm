import utils
import nimfm/dataset, nimfm/model, nimfm/regularizer, nimfm/tensor
import nimfm/optimizer/pgd


when isMainModule:
  var XTr, XTe: CSRDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("dataset/ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("dataset/ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var reg = newSquaredL12()
  let beta = 1e-5
  let gamma = 1e-5

  var fm = newFactorizationMachine(task=regression)
  var pgd = newPGD(
    maxIter=100, beta=beta, alpha0=1e-10, alpha=1e-10,
    gamma=gamma, reg=reg)
  pgd.fit(Xtr, yTr, fm)

  echo("L1 norm of the interaction matrix: ", norm(matmul(fm.P[0].T, fm.P[0]), 1))
  echo("Number of used interactions: ", countInteractions(fm.P[0].T))
  echo("Number of used features: ", countFeatures(fm.P[0].T))
  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
  #[
  L1 norm of the interaction matrix: 269.9612424535608
  Number of used interactions: 3124578
  Number of used features: 2701
  Train RMSE: 1.045277114672563
  Test RMSE: 1.037980185747495

  real	0m55.505s
  user	0m55.381s
  sys	0m0.053s
  ]#