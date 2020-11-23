type
  BaseOptimizerObj = object of RootObj
    verbose*: int
    tol*: float64
    maxIter*: int
    alpha0*: float64
    alpha*: float64
    beta*: float64

  BaseOptimizer* = ref object of BaseOptimizerObj

  BaseCSCOptimizer* = ref object of BaseOptimizer

  BaseCSROptimizer* = ref object of BaseOptimizer
