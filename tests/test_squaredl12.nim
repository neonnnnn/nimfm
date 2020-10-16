import unittest, sequtils
import nimfm/regularizers/regularizers
import regularizers/squaredl12_slow, random


suite "Test proximal operator of SquaredL12":
  let
    d = 100

  test "Test proximal operation":
    var
      reg = newSquaredL12()
    
    reg.initSGD(nFeatures = d, nComponents=4, degree=2)
    var q = newSeqWith(d, 0.0)
    for i in 0..<1000:
      for j in 0..<d:
        q[j] = float(rand(400))/100-2.0
      for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4]:
        var p1 = q
        var candidates = @[0,1,2,3,4,5]
        proxSquaredL12(p1, lam, candidates)
        var p2 = q
        proxSquaredL12Slow(p2, lam, 2)
        for j in 0..<6:
          check abs(p1[j] - p2[j]) < 1e-10