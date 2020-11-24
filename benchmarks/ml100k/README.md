# Benchmarks on MoviLens 100K dataset
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) is a dataset
for the movie recommendation task. It has 943 users, 1,682 items, 
and 100,000 ratings. It also provides some additional information and
factorization machines can leverage them.

We provide codes for some baseline methods [1], factorization machines [2],
and higher-order factorization machines [3] on the MovieLens 100K dataset.
The baseline methods implemented in this benchmark are represented as the 
factorization machines by changing input features and
hyperparameter setting (see [2]).

In addition, this benchmark provides factorization machines with sparse regularization [4,5,6,7].

## Dependencies
 - [zip library in Nim](https://github.com/nim-lang/zip)

## Usage
1. Compile and run `make_ml100k_dataset.nim` with `-d:ssl` option:


    nim c --run --d:ssl make_ml100k_dataset.nim

 Then, `ml-100k.zip` will be downloaded and uncompressed, and following
 files will be created in `dataset` directory:
   - `ml-100k_user_item_all.svm`
   - `ml-100k_user_item_train.svm`
   - `ml-100k_user_item_test.svm`
   - `ml-100k_user_item_feature_all.svm`
   - `ml-100k_user_item_feature_train.svm`
   - `ml-100k_user_item_feature_test.svm`

  First three files are svmlight format dataset files for 
  `matrix_factorization.nim` and `user_item_bias.nim`. 
  `matrix_factorization.nim` provides matrix factorization (MF) (a.k.a latent
  factor (feature) model) methods [1]. `user_item_bias.nim` provides the linear
  regression with user-id and item-id as input. It predicts the rating as
  overall_bias + user_bias + item_bias.
  
  If you want to run field-aware factorization machines examples (`ffm_sgd.nim` and `ffm_adagrad.nim`), please compile and rune `make_ml100k_dataset_field.nim`.

2. Compile other nim files with `-d:release` and `-d:danger`, 
   and run them. For example,

   nim c --run --d:release --d:danger matrix_factorization.nim

  `factorization_machine.nim`, 
  `factorization_machine_sgd.nim`, `factorization_machines_adagrad.nim`, `higher_order_factorization_machine.nim`,
  and `linear_model.nim` use not only user-id and item-id but also
   - age, occupation, sex, and zipcode of user (dimension: 49),
   - released year and genre of item (dimension: 29).

   For more details about feature encoding, please see [3].

   `factorization_machine_stream.nim` and `factorization_machine_sgd_stream.nim` are examples using our binary data format.
   Before running them, you must run `convert.nim`.
   It outputs binary versions of `ml-100k_user_item_feature_train.svm` and `ml-100k_user_item_feature_test.svm`.

   `sparse*` provides factorization machines with sparse regularization [4,5,6,7].

   When compiling `factorization_machine_sgd_multi`, `factorization_machine_stream_multi`, `factorization_machine_adagrad_multi`, `ffm_sgd`, or `ffm_adagrad`, use `--threads:on` flag. They use multiple threads.

## References
1. Y. Koren. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In KDD, pp. 426--434, 2008.

2. S. Rendle. Factorization machines. In ICDM, pp. 995--1000, 2010.

3. M. Blondel, A. Fujino, N. Ueda, M. Ishihata. Higher-order factorization machines. In NeurIPS, pp. 3351--3359, 2016.

4. Z. Pan, E. Chen, Q. Liu, T. Xu, H. Ma, and H. Lin. Sparse factorization machines for click-through rate prediction. In ICDM, pp. 400--409, 2016.

5. J Xu, K Lin, P. N. Tan, and J. Zhou. Synergies that matter: Efficient interaction selection via sparse factorization machine. In SDM, pp. 1008-–0116, 2016.

6. H. Zhao, Q. Yao, J. Li, Y. Song, and D. L. Lee. Meta-graph based recommendation fusion over heterogeneous information networks. In KDD, pp. 635–-644, 2017

7. K. Atarashi, S. Oyama, and M. Kurihara. Factorization machines with regularization for sparse feature interactions. preprint.
