# nimfm
A library for factorization machines in [Nim](https://nim-lang.org/).

[![Actions Status](https://github.com/neonnnnn/nimfm/workflows/Build/badge.svg)](https://github.com/neonnnnn/nimfm/actions)


Factorization machines (FMs)[1, 2] are machine learning models using second-order
feature combinations (e.g., x1 * x2, x2 * x5) efficiently.

nimfm provides

 - Not-only second-order but also higher-order factorization machines [3].
 - Coordinate descent (a.k.a alternative least squares) solver.
 - Stochastic gradient descent solver with some step-size scheduling methods [4].
 - Greedy coordinate descent [5] and Hazan's (Frank-Wolfe) algorithm [6] (with some heuristics) solvers for convex factorization machines.
 - Various loss functions: Squared, Huber, SquaredHinge, and Logistic.
 - Binary file for end users.

## Data format
nimfm uses its own data type for datasets: `CSRDataset` and `CSCDataset` and provides procs for loading **[libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)/[svmlight](http://svmlight.joachims.org/) format** file as such datasets.
`CSRDataset` is for `SGD` solver and `CSCDataset` is for `CD`, `GreedyCD`, and `Hazan` solvers.
The two-dimensional sequence `seq[seq[float64]]` is easily transformed to such
datasets by `toCSR` and `toCSC`.


## Installation for Nim users
 Install by [nimble](https://github.com/nim-lang/nimble/):
 
 
    nimble install https://github.com/neonnnnn/nimfm


## Installation for end users
 1. Download the source codes by
 
 
    git clone https://github.com/neonnnnn/nimfm.git


 2. Make:


    nimble make

Then, nimfm binary will be created in the ./bin directory.
`.bin/nimfm --help` shows the usage.

## References

1. S. Rendle. Factorization machines. In ICDM, pp. 995--1000, 2010.

2. S. Rendle. Factorization machines with libfm. ACM Transactions on Intelligent Systems and Technology, 3(3):57--78, 2012.

3. M. Blondel, A. Fujino, N. Ueda, M. Ishihata. Higher-order factorization machines. In NeurIPS, pp. 3351--3359, 2016.

4. L. Bottou. Stochastic gradient descent tricks. Neural Networks, Tricks of the Trade, Reloaded, pp. 430–445, 
 Lecture Notes in Computer Science (LNCS 7700), Springer, 2012.

5. M. Blondel, A. Fujino, and N. Ueda. Convex factorization machines. In ECML-PKDD, pp. 19--35, 2015.

6. M. Yamada, W. Lian, A. Goyal, J. Chen, K. Wimalawarne, S. A. Khan, S. Kaski, H. Mamitsuka, and Y. Chang.
   Convex factorization machine for toxicogenomics prediction. In KDD, pp. 1215--1224, 2017.

## Authors
 - Kyohei Atarashi, 2020-present
