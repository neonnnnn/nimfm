import nimfm


when isMainModule:
  var XTr, XTe: CSRDataset  # Use CSRDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var fm = newFactorizationMachine(task=regression)
  var adagrad = newAdaGraD(eta0=0.1, maxIter=100, beta=1e-3,
                           alpha0=1e-10, alpha=1e-10)
  adagrad.fit(Xtr, yTr, fm)

  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
