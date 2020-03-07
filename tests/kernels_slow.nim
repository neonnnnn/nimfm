import nimfm/tensor
import utils


proc anovaSlow*(X: Matrix,  P: Tensor, 
                i, degree, order, s, d, m:int): float64 = 
  result = 0.0
  for indices in comb(d+m, degree):
    var prod = 1.0
    for j in indices:
      prod *= P[order, s, j]
      if j < d:
        prod *= X[i, j]
    result += prod
