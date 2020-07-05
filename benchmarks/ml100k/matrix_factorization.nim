import nimfm


when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_train.svm", XTr, yTr, nFeatures=2625)
  loadSVMLightFile("ml-100k_user_item_test.svm", XTe, yTe, nFeatures=2625)

  var fm = newFactorizationMachine(task=regression, beta=3e-4, alpha=1e-10, alpha0=1e-10)
  var cd = newCD(maxIter=100)
  cd.fit(Xtr, yTr, fm)

  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
