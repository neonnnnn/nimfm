import utils
import nimfm/dataset, nimfm/model, nimfm/regularizer, nimfm/tensor, nimfm/tensor
import nimfm/optimizer/pcd


when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("dataset/ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("dataset/ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var reg = newSquaredL12()
  let beta = 1e-5
  let gamma = 1e-5

  var fm = newFactorizationMachine(
    task=regression, warmStart=false, degree=2)
  var optim = newPCD(
    maxIter=100, verbose=1, beta=beta, alpha0=1e-10, alpha=1e-5,
    gamma=gamma, reg=reg)
  optim.fit(Xtr, yTr, fm)

  echo("L1 norm of the interaction matrix: ", norm(matmul(fm.P[0].T, fm.P[0]), 1))
  echo("Number of used interactions: ", countInteractions(fm.P[0].T))
  echo("Number of used features: ", countFeatures(fm.P[0].T))
  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))

  #[
  L1 norm of the interaction matrix: 2445.891815236551
  Number of used interactions: 259713
  Number of used features: 1316
  Train RMSE: 0.8499253341126667
  Test RMSE: 0.9190020120369393

  real	0m29.862s
  user	0m29.613s
  sys	0m0.149s
  ]#
 