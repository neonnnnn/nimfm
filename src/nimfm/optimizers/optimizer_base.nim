type
  BaseOptimizerObj = object of RootObj
    verbose*: int
    tol*: float64
    maxIter*: int

  BaseOptimizer* = ref object of BaseOptimizerObj

  BaseCSCOptimizer* = ref object of BaseOptimizer

  BaseCSROptimizer* = ref object of BaseOptimizer
