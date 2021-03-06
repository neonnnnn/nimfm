import nimfm/dataset, nimfm/model
import nimfm/optimizer/hazan, nimfm/optimizer/greedy_cd


when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("dataset/ml-100k_user_item_train.svm",
                    XTr, yTr, nFeatures=2625)
  loadSVMLightFile("dataset/ml-100k_user_item_test.svm",
                    XTe, yTe, nFeatures=2625)

  var cfm = newConvexFactorizationMachine(
    task=regression, maxComponents=50, ignoreDiag=true, fitLinear=true,
    fitIntercept=true, warmStart=false)

  var optimGCD = newGreedyCD(
    maxIter=30, maxIterInner=10, maxIterPower=100,
    beta=1e-4, alpha0=1e-8, alpha=1e-8,
    refitFully=false, nRefitting=10, verbose=1)
  echo("Training CFM by GreedyCD.")
  optimGCD.fit(Xtr, yTr, cfm)
  echo("Train RMSE: ", cfm.score(Xtr, yTr))
  echo("Test RMSE: ", cfm.score(Xte, yTe))
  echo()

  var optimHazan = newHazan(
    maxIter=100, maxIterPower=100, optimal=true, verbose=1,
    eta=600, nTol=100)
  echo("Training CFM by Hazan's Algorothm.")
  optimHazan.fit(Xtr, yTr, cfm)
  echo("Train RMSE: ", cfm.score(Xtr, yTr))
  echo("Test RMSE: ", cfm.score(Xte, yTe))