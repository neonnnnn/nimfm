import nimfm/tensor/tensor
import comb
import math


proc anovaSlow*(x, p: Vector, degree: int): float64 = 
  result = 0.0
  let d = len(x)
  if d != len(p):
    raise newException(ValueError, "len(x) != len(p).")
  for indices in comb(d, degree):
    var prod = 1.0
    for j in indices:
      prod *= p[j]*x[j]
    result += prod


proc anovaSlow*(X: Matrix,  P: Matrix, 
                i, degree, s, d, m:int): float64 = 
  result = 0.0
  for indices in comb(d+m, degree):
    var prod = 1.0
    for j in indices:
      prod *= P[s, j]
      if j < d:
        prod *= X[i, j]
    result += prod


proc polySlow*(X, P: Matrix, i, degree, s, d, m: int): float64 =
  result = 0.0
  for j in 0..<d:
    result += X[i, j] * P[s, j]
  for j in 0..<m:
    result += X[i, d+j] * P[s, d+j]
  result = result^degree
