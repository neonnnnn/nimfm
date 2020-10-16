import nimfm, utils
import strformat


when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var reg = newL1()
  let beta = 1e-5
  for gamma in [1e-5, 2e-5, 4e-5, 8e-5, 16e-5]:
    var fm = newFactorizationMachine(
      task=regression, warmStart=false, degree=2)
    var cd = newPCD(
      maxIter=100, verbose=0, beta=beta, alpha0=1e-10, alpha=1e-5,
      gamma=gamma, reg=reg)
    cd.fit(Xtr, yTr, fm)

    echo(fmt"gamma = {gamma}")
    echo("L1 norm of the interaction matrix: ", norm(matmul(fm.P[0].T, fm.P[0]), 1))
    echo("Number of used interactions: ", countInteractions(fm.P[0].T))
    echo("Number of used features: ", countFeatures(fm.P[0].T))
    echo("Train RMSE: ", fm.score(Xtr, yTr))
    echo("Test RMSE: ", fm.score(Xte, yTe))
    echo()

    #[
    gamma = 1e-05
    L1 norm of the interaction matrix: 1309723.455536645
    Number of used interactions: 3109179
    Number of used features: 2587
    Train RMSE: 0.4609503168064058
    Test RMSE: 1.114549341939459

    gamma = 2e-05
    L1 norm of the interaction matrix: 581084.7126608792
    Number of used interactions: 2806571
    Number of used features: 2583
    Train RMSE: 0.5822531018523225
    Test RMSE: 1.054355837686803

    gamma = 4e-05
    L1 norm of the interaction matrix: 151744.6942830395
    Number of used interactions: 2332458
    Number of used features: 2541
    Train RMSE: 0.7349862895593922
    Test RMSE: 0.956525304081841

    gamma = 8.000000000000001e-05
    L1 norm of the interaction matrix: 12329.76350545176
    Number of used interactions: 1508208
    Number of used features: 2340
    Train RMSE: 0.8417132217173198
    Test RMSE: 0.9303608420224291

    gamma = 0.00016
    L1 norm of the interaction matrix: 81.99300072742886
    Number of used interactions: 2080
    Number of used features: 65
    Train RMSE: 0.9103939645491346
    Test RMSE: 0.9302782353900708
    ]#