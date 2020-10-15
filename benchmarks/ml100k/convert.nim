import nimfm


when isMainModule:
  convertSVMLightFile("ml-100k_user_item_feature_train.svm",
                      "ml-100k_user_item_feature_train_samples",
                      "ml-100k_train_labels")
  convertSVMLightFile("ml-100k_user_item_feature_test.svm",
                      "ml-100k_user_item_feature_test_samples",
                      "ml-100k_test_labels")
  transposeFile("ml-100k_user_item_feature_train_samples",
                "ml-100k_user_item_feature_train_samples_csc",
                cachesize=5)
  transposeFile("ml-100k_user_item_feature_train_samples_csc",
                "ml-100k_user_item_feature_train_samples_csc_transpose",
                cachesize=5)