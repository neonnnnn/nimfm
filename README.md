# nimfm
nimfm: a library for factorization machines in Nim.

Factorization machines (FMs) are machine learning models using second-order
feature combinations (e.g., x1 * x2, x2 * x5) efficiently.

nimfm provides

 - Not-only second-order but also higher-order factorization machines.
 - Coordinate descent (a.k.a alternative least squares) solver.
 - Stochastic gradient descent solver with some step-size scheduling methods.
 - Various loss functions: Squared, SquaredHinge, and Logistic.
 - Binary file for end users.

## Data format
nimfm uses its own data type for datasets: `CSRDataset` and `CSCDataset`,
and provides procs for loading **libsvm/svmlight format** file as such datasets.
The two-dimensional sequence `seq[seq[float64]]` is easily transformed to such
datasets by `toCSR` and `toCSC`.


## Installation for Nim users
 1. Download the source codes by
 
 
    git clone https://github.com/neonnnnn/nimfm.git

 2. Install by nimble:
    

    cd nimfm

    nimble install


## Installation for end users
 1. Download the source codes by
 
 
    git clone https://github.com/neonnnnn/nimfm.git


 2. Make:


    nimble make

Then, nimfm binary will be created in the ./bin directory.
`.bin/nimfm --help` shows the usage.


## Authors
 - Kyohei Atarashi, 2020-present
