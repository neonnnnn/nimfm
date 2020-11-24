import nimfm


when isMainModule:
  convertSVMLightFile("dataset/ml-100k_user_item_feature_train.svm",
                      "dataset/ml-100k_user_item_feature_train_samples",
                      "dataset/ml-100k_train_labels")
  convertSVMLightFile("dataset/ml-100k_user_item_feature_test.svm",
                      "dataset/ml-100k_user_item_feature_test_samples",
                      "dataset/ml-100k_test_labels")
  transposeFile("dataset/ml-100k_user_item_feature_train_samples",
                "dataset/ml-100k_user_item_feature_train_samples_csc",
                cachesize=5)
  transposeFile("dataset/ml-100k_user_item_feature_train_samples_csc",
                "dataset/ml-100k_user_item_feature_train_samples_csc_transpose",
                cachesize=5)