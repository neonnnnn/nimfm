import nimfm

when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_train.svm",
                    XTr, yTr, nFeatures=2625)
  loadSVMLightFile("ml-100k_user_item_test.svm",
                    XTe, yTe, nFeatures=2625)

  var cfm = newConvexFactorizationMachine(
    task=regression, beta=1e-4, alpha0=1e-8, alpha=1e-8,
    maxComponents=50, ignoreDiag=true, fitLinear=true,
    fitIntercept=true, eta=600, warmStart=true)

  var gcd = newGreedyCD(
    maxIter=10, maxIterInner=10, maxIterPower=100,
    refitFully=false, nRefitting=100, verbose=1)

  var hazan = newHazan(maxIter=100, maxIterPower=100, optimal=true, verbose=1,
                       nTol=100)
  #hazan.fit(Xtr, yTr, cfm)
  gcd.fit(Xtr, yTr, cfm)
  echo("Train RMSE: ", cfm.score(Xtr, yTr))
  echo("Test RMSE: ", cfm.score(Xte, yTe))