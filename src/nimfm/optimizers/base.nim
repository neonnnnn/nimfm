import ../dataset, ../factorization_machine

type
  BaseOptimizerObj = object of RootObj
    verbose*: bool
    tol*: float64
    maxIter*: int

  BaseOptimizer* = ref object of BaseOptimizerObj

  BaseCSCOptimizer* = ref object of BaseOptimizer

  BaseCSROptimizer* = ref object of BaseOptimizer


proc fit*(self: BaseCSCOptimizer, X: CSCDataset, y: seq[float64],
          fm: FactorizationMachine) = discard


proc fit*(self: BaseCSROptimizer, X: CSRDataset, y: seq[float64],
          fm: FactorizationMachine) = discard
