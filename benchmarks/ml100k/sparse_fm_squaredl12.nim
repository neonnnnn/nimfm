import nimfm, utils
import strformat


when isMainModule:
  var XTr, XTe: CSCDataset
  var yTr, yTe: seq[float64]
  loadSVMLightFile("ml-100k_user_item_feature_train.svm",
                    XTr, yTr, nFeatures=2703)
  loadSVMLightFile("ml-100k_user_item_feature_test.svm",
                    XTe, yTe, nFeatures=2703)

  var reg = newSquaredL12()
  let beta = 1e-5
  for gamma in [0.25e-5, 0.5e-5, 1e-5, 2e-5, 4e-5]:
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
  gamma = 2.5e-06
  L1 norm of the interaction matrix: 10705.65361680202
  Number of used interactions: 1038370
  Number of used features: 2185
  Train RMSE: 0.804163491026079
  Test RMSE: 0.929679006965786

  gamma = 5e-06
  L1 norm of the interaction matrix: 5209.316795226093
  Number of used interactions: 608759
  Number of used features: 1831
  Train RMSE: 0.8271503104946665
  Test RMSE: 0.9230673890845948

  gamma = 1e-05
  L1 norm of the interaction matrix: 2445.891815236551
  Number of used interactions: 259713
  Number of used features: 1316
  Train RMSE: 0.8499253341126667
  Test RMSE: 0.9190020120369393

  gamma = 2e-05
  L1 norm of the interaction matrix: 823.0821924538147
  Number of used interactions: 65933
  Number of used features: 697
  Train RMSE: 0.8771468870630396
  Test RMSE: 0.9192673067881156

  gamma = 4e-05
  L1 norm of the interaction matrix: 188.0835552498489
  Number of used interactions: 8242
  Number of used features: 222
  Train RMSE: 0.8976343460718412
  Test RMSE: 0.9243847426000633
  ]#