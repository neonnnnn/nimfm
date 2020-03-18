import nimfm


when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var cfm = newConvexFactorizationMachine(
    task=regression, beta=1e-4, alpha0=1e-8, alpha=1e-8,
    maxComponents=50, ignoreDiag=true)
  var gcd = newGreedyCoordinateDescent(
    maxIter=100, maxIterPower=1000, tolPower=1e-8)
  gcd.fit(Xtr, yTr, cfm)

  echo("Train RMSE: ", cfm.score(Xtr, yTr))
  echo("Test RMSE: ", cfm.score(Xte, yTe))