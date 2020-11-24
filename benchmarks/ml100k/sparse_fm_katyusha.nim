import utils
import nimfm/dataset, nimfm/model, nimfm/regularizer, nimfm/tensor
import nimfm/optimizer/katyusha


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
  var optim = newKatyusha(
    eta=0.1, maxIter=30, beta=beta, alpha0=1e-10,
    alpha=1e-10, gamma=gamma, reg=reg)
  optim.fit(Xtr, yTr, fm)

  echo("L1 norm of the interaction matrix: ", norm(matmul(fm.P[0].T, fm.P[0]), 1))
  echo("Number of used interactions: ", countInteractions(fm.P[0].T))
  echo("Number of used features: ", countFeatures(fm.P[0].T))
  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))

  #[
  Number of used interactions: 3651753
  Number of used features: 2703
  Train RMSE: 0.9371118541076972
  Test RMSE: 0.9422047401514687

  real	0m56.922s
  user	0m56.695s
  sys	0m0.080s
  ]#