import nimfm/dataset, nimfm/model
#import nimfm/optimizer/sgd_ffm # single-threading fit
import nimfm/optimizer/sgd_ffm_multi

when isMainModule:
  var XTr, XTe: CSRFieldDataset  # Use CSRFieldDataset for SGD solver
  var yTr, yTe: seq[float64]
  let scheduling: SchedulingKind = optimal
  loadFFMFile("dataset/ml-100k_user_item_feature_train.ffm",
               XTr, yTr, nFeatures=2703, nFields=8)
  loadFFMFile("dataset/ml-100k_user_item_feature_test.ffm",
               XTe, yTe, nFeatures=2703, nFields=8)

  var ffm = newFieldAwareFactorizationMachine(task=regression)
  var optim = newSGD(eta0=0.01, scheduling=scheduling, maxIter=100,
                     beta=1e-3, alpha0=1e-10, alpha=1e-10)
  # optim.fit(XTr, yTr, ffm) # single-threading fit
  optim.fit(XTr, yTr, ffm, maxThreads=4)
  echo("Train RMSE: ", ffm.score(XTr, yTr))
  echo("Test RMSE: ", ffm.score(Xte, yTe))
