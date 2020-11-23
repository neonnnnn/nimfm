import nimfm/dataset, nimfm/model
import nimfm/optimizer/sgd_multi


when isMainModule:
  var XTr, XTe: CSRDataset  # Use CSRDataset for SGD solver
  var yTr, yTe: seq[float64]
  let scheduling: SchedulingKind = optimal
  loadSVMLightFile("dataset/ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("dataset/ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)
  var fm = newFactorizationMachine(task=regression)
  var optim = newSGD(eta0=0.01, scheduling=scheduling, maxIter=100, tol = -10,
                    beta=1e-3, alpha0=1e-10, alpha=1e-10, shuffle=true)
  optim.fit(XTr, yTr, fm, maxThreads=4)

  echo("Train RMSE: ", fm.score(XTr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
