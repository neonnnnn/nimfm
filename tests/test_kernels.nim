import unittest, random, sequtils
import nimfm/kernels, nimfm/dataset, nimfm/tensor/tensor
import kernels_slow


suite "Test ANOVA Kernel Computation":
  let
    n = 20
    d = 10
    k = 10
    nAug = 3
  # create data matrix and seq
  var XSeq = newSeqWith(n, newSeqWith(d, 0.0))
  randomize(42)
  for i in 0..<n:
    for j in 0..<d:
      XSeq[i][j] = rand(1.0)
  
  let P = randomNormal([1, k, d+nAug])

  let XCSR = toCSRDataset(XSeq)
  let XCSC = toCSCDataset(XSeq)
  var K = zeros([n, d+1])
  let XMat = toMatrix(XSeq)

  test "Test ANOVA Kernel for CSC":
    for m in 0..<nAug+1:
      XCSC.addDummyFeature(1.0, m)
      for degree in 2..<6:
        for s in 0..<k:
          anova(XCSC, P[0], K, degree, s)
          for i in 0..<n:
            let expect = anovaSlow(XMat, P[0], i, degree, s, d, m)
            check abs(K[i, degree] -  expect) < 1e-6
      XCSC.removeDummyFeature(m)
    
  test "Test ANOVA Kernel for CSR":
    for m in 0..<nAug+1:
      XCSR.addDummyFeature(1.0, m)
      for degree in 2..<6:
        for s in 0..<k:
          anova(XCSR, P[0], K, degree, s)
          for i in 0..<n:
            let expect = anovaSlow(XMat, P[0], i, degree, s, d, m)
            check abs(K[i, degree] - expect) < 1e-6
      XCSR.removeDummyFeature(m)