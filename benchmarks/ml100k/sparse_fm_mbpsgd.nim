import nimfm, utils


when isMainModule:
  var XTr, XTe: CSRDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)
  var reg = newSquaredL12()
  let beta = 1e-5
  let gamma = 1e-5

  var fm = newFactorizationMachine(task=regression)
  var mbpsgd = newMBPSGD(
    maxIter=30, beta=beta, alpha0=1e-10, alpha=1e-10,
    gamma=gamma, reg=reg)
  mbpsgd.fit(Xtr, yTr, fm)

  echo("L1 norm of the interaction matrix: ", norm(matmul(fm.P[0].T, fm.P[0]), 1))
  echo("Number of used interactions: ", countInteractions(fm.P[0].T))
  echo("Number of used features: ", countFeatures(fm.P[0].T))
  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
  
  #[
  Number of used interactions: 2190654
  Number of used features: 2147
  Train RMSE: 0.9453493094003254
  Test RMSE: 0.9487633539711476

  real	0m30.019s
  user	0m29.920s
  sys	0m0.057s
  ]#