import nimfm/dataset, nimfm/model
import nimfm/optimizer/sgd


when isMainModule:
  var XTr = newStreamCSRDataset("dataset/ml-100k_user_item_feature_train_samples", cacheSize=5)
  var XTe = newStreamCSRDataset("dataset/ml-100k_user_item_feature_test_samples")
  var yTr = loadStreamLabel("dataset/ml-100k_train_labels")
  var yTe = loadStreamLabel("dataset/ml-100k_test_labels")
  var fm = newFactorizationMachine(task=regression)
  let scheduling: SchedulingKind = constant
  var optim = newSGD(eta0=0.01, scheduling=scheduling, maxIter=100,
                     beta=1e-3, alpha0=1e-10, alpha=1e-10, shuffle=false)

  optim.fit(Xtr, yTr, fm)

  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))
