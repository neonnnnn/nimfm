import nimfm/dataset, nimfm/model
import nimfm/optimizer/cd


when isMainModule:
  var XTr = newStreamCSCDataset("dataset/ml-100k_user_item_feature_train_samples_csc",
                                 cacheSize=5)
  var XTe = newStreamCSRDataset("dataset/ml-100k_user_item_feature_test_samples")
  var yTr = loadStreamLabel("dataset/ml-100k_train_labels")
  var yTe = loadStreamLabel("dataset/ml-100k_test_labels")
  var fm = newFactorizationMachine(task=regression)
  var optim = newCD(maxIter=100,  beta=1e-3, alpha0=1e-10, alpha=1e-10)
  optim.fit(Xtr, yTr, fm)

  echo("Train RMSE: ", fm.score(Xtr, yTr))
  echo("Test RMSE: ", fm.score(Xte, yTe))