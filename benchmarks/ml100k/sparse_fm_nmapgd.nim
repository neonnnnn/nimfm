import utils
import nimfm/dataset, nimfm/model, nimfm/regularizer, nimfm/tensor
import nimfm/optimizer/nmapgd

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
  var optim = newNMAPGD(
    maxIter=100, beta=beta, alpha0=1e-10, alpha=1e-10,
    gamma=gamma, reg=reg, eta=0.8, sigma=0.1)
  optim.fit(Xtr, yTr, fm)
  
  echo("L1 norm of the interaction matrix: ", norm(matmul(fm.P[0].T, fm.P[0]), 1))
  echo("Number of used interactions: ", countInteractions(fm.P[0].T))
  echo("Number of used features: ", countFeatures(fm.P[0].T))
  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
  
  #[
  L1 norm of the interaction matrix: 419.4712207824275
  Number of used interactions: 1600921
  Number of used features: 2563
  Train RMSE: 0.9578042428692431
  Test RMSE: 0.9569467738334844

  real	0m49.066s
  user	0m48.916s
  sys	0m0.063s
  ]#