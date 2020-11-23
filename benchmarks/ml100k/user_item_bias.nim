import nimfm/dataset, nimfm/model
import nimfm/optimizer/cd



when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("dataset/ml-100k_user_item_train.svm", XTr, yTr, nFeatures=2625)
  loadSVMLightFile("dataset/ml-100k_user_item_test.svm", XTe, yTe, nFeatures=2625)

  var fm = newFactorizationMachine(task=regression, degree=1)
  var optim = newCD(maxIter=100, alpha=1e-10, alpha0=1e-10)
  optim.fit(Xtr, yTr, fm)

  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
