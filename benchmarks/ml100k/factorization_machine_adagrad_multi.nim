import nimfm/dataset, nimfm/model, nimfm/loss
import nimfm/optimizer/adagrad_multi


when isMainModule:
  var XTr, XTe: CSRDataset  # Use CSRDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("dataset/ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("dataset/ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var fm = newFactorizationMachine(task=regression)
  var optim = newAdaGrad(eta0=0.1, maxIter=100, beta=1e-3,
                         alpha0=1e-10, alpha=1e-10)
  optim.fit(XTr, yTr, fm, maxThreads=4)

  echo("Train RMSE: ", fm.score(XTr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
