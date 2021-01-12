# nimfm
A library for factorization machines in [Nim](https://nim-lang.org/).

[![Actions Status](https://github.com/neonnnnn/nimfm/workflows/Build/badge.svg)](https://github.com/neonnnnn/nimfm/actions)


Factorization machines (FMs)[1,2] are machine learning models using second-order feature combinations (e.g., x1 * x2, x2 * x5) efficiently.

nimfm provides

 - Not-only second-order but also higher-order factorization machines [3].
 - Coordinate descent (a.k.a alternative least squares) solver.
 - Stochastic gradient descent solver with some step-size scheduling methods [4] and AdaGrad solver [5].
 - Greedy coordinate descent [6] and Hazan's (Frank-Wolfe) algorithm [7] (with some heuristics) solvers for convex factorization machines.
 - Some sparse regularizers [8,9,10,11] for feature selection and feature interaction selection, and various optimizers for such regularizers [11,12,13,14].
 - Field-aware factorization machines [15] and AdaGrad/SGD solver for them.
 - Various loss functions: Squared, Huber, SquaredHinge, and Logistic.
 - Binary file for end users.

## Data format
### For `FactorizationMachine` and `ConvexFactorizationMachine`
nimfm uses its own data type for datasets, `CSRDataset` and `CSCDataset`, and provides procs for loading **[libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)/[svmlight](http://svmlight.joachims.org/) format** file as such datasets.
`CSRDataset` is for `SGD`, `Adagrad`, `PSGD`, `Katyusha`, `PGD`, `MBPSGD`, `FISTA` and `NMAPGD` solvers.
`CSCDataset` is for `CD`, `GreedyCD`, `Hazan`, `PCD`, and `PBCD` solvers.

In addition, nimfm provides binary data formats: `StreamCSRDataset` and `StreamCSCDataset`.
They have only parts of a dataset in main memory and therefore are useful for a very large-scale dataset.
`convertSVMLightFile` converts a svmlight format file to a `StreamCSRDataset` format file. 
`transposeFile` converts a `StreamCSRDataset` format file to a `StreamCSCDataset` format file.

The two-dimensional sequence `seq[seq[float64]]` can be easily transformed to such datasets by `toCSR` and `toCSC`.

### For `FieldAwareFactorizationMachine`
For field-aware factorization machines, nimfm provides `CSRFieldDataset`, `CSCFieldDataset`, and procs for loading **[libffm](https://www.csie.ntu.edu.tw/~cjlin/libffm/) format** file as such datasets.
Binary formats for field-aware datasets, `StreamCSRFieldDataset` and `StreamCSCFieldDataset`, are also supported.
Note that currently (version 0.3.0) nimfm provides no solver using `CSCFieldDataset`.

## Sparse regularizers
nimfm provides some sparse regularizers [8,9,10,11] and various optimizers for such regularizers [11,12,13,14].
The following table shows whether each optimizer can be used for each regularizer or not.

|Optimizer \ Regularizer | `L1` [8] | `L21` [9,10] | `SquaredL12` [11] | `SquaredL21` [11]| `OmegaTI` [11]| `OmegaCS` [11]|
|------------------------|------|-------|--------------|--------------|-----------|-----------|
|`PCD` [11]|**Yes**|No|**Yes**|No|**Yes**|No|
|`PBCD` [11]|**Yes**|**Yes**|No|**Yes**|No|**Yes**|
|`PGD`|**Yes**|**Yes**|**Yes**|**Yes**|No|No|
|`FISTA` [12]|**Yes**|**Yes**|**Yes**|**Yes**|No|No|
|`NMAPGD` [13]|**Yes**|**Yes**|**Yes**|**Yes**|No|No|
|`PSGD`|**Yes**|**Yes**|**Yes**|**Yes**|No|No|
|`MBPSGD` (MB = MiniBatch)|**Yes**|**Yes**|**Yes**|**Yes**|No|No|
|`Katyusha` [14]|**Yes**|**Yes**|**Yes**|**Yes**|No|No|

`L21`, `SquaredL21`, and `OmegaCS` are for feature selection and others are for feature interaction selection.
Note that `PSGD` for `SquaredL12` and `SquaredL21` might be slow when a dataset is sparse.

## Multi-threading
Currently (version 0.3.0), `SGD` and `AdaGrad` solvers for `FactorizationMachine` and `FieldAwareFactorizationMachine` can run with multi-threading.
They update model parameters **asynchronously**, so their results are different from those of single-thearding version.


## Installation for Nim users
 Install by [nimble](https://github.com/nim-lang/nimble/):
 
 
    nimble install nimfm


## Installation for end users
 1. Download the source codes by
 
 
    git clone https://github.com/neonnnnn/nimfm.git


 2. Make:


    nimble make

Then, `nimfm`, `nimfm_cfm`, and `nimfm_sparsefm` binary will be created in the ./bin directory.
`nimfm` is for conventional factorization machines, `nimfm_cfm` is for convex factorization machines, and `nimfm_sparsefm` is for factorization machines with sparse regularization.
`./bin/[binary-name] --help` shows the usage.

## How to use nimfm?
Please see examples at `benchmarks` directory.
If you are familiar with **[scikit-learn](https://scikit-learn.org/stable/)**, you probably will be able to use nimfm well.

## References

1. S. Rendle. Factorization machines. In ICDM, pp. 995--1000, 2010.

2. S. Rendle. Factorization machines with libfm. ACM Transactions on Intelligent Systems and Technology, 3(3):57--78, 2012.

3. M. Blondel, A. Fujino, N. Ueda, M. Ishihata. Higher-order factorization machines. In NeurIPS, pp. 3351--3359, 2016.

4. L. Bottou. Stochastic gradient descent tricks. Neural Networks, Tricks of the Trade, Reloaded, pp. 430–445, Lecture Notes in Computer Science (LNCS 7700), Springer, 2012.

5. J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121-–2159, 2011. 

6. M. Blondel, A. Fujino, and N. Ueda. Convex factorization machines. In ECML-PKDD, pp. 19--35, 2015.

7. M. Yamada, W. Lian, A. Goyal, J. Chen, K. Wimalawarne, S. A. Khan, S. Kaski, H. Mamitsuka, and Y. Chang. Convex factorization machine for toxicogenomics prediction. In KDD, pp. 1215--1224, 2017.

8. Z. Pan, E. Chen, Q. Liu, T. Xu, H. Ma, and H. Lin. Sparse factorization machines for click-through rate prediction. In ICDM, pp. 400--409, 2016.

9. J Xu, K Lin, P. Tan, and J. Zhou. Synergies that matter: Efficient interaction selection via sparse factorization machine. In SDM, pp. 1008-–0116, 2016.

10. H. Zhao, Q. Yao, J. Li, Y. Song, and D. L. Lee. Meta-graph based recommendation fusion over heterogeneous information networks. In KDD, pp. 635–-644, 2017

11. K. Atarashi, S. Oyama, and M. Kurihara. Factorization machines with regularization for sparse feature interactions. https://arxiv.org/abs/2010.09225

12. A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1):183-–202, 2009.

13. H. Li and Z. Lin. Accelerated proximal gradient methods for nonconvex programming. In NeurIPS, pp. 379-–387, 2015.

14. Z. Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. Journal of Machine Learning Research, 18(1):8194–-8244, 2017.

15. Y. Juan, Y. Zhuang, W-S. Chin, and C-J. Lin: Field-aware factorization machines for CTR prediction. In SIGIR, pp. 43--50, 2016.

## Authors
 - Kyohei Atarashi, 2020-present
