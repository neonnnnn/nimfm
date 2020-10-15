import unittest
import utils, optimizers/greedy_cd_slow, models/cfm_slow
import nimfm/loss, nimfm/dataset, nimfm/tensor/tensor
import nimfm/models/convex_factorization_machine, nimfm/models/fm_base
from nimfm/models/factorization_machine import FitLowerKind
import nimfm/optimizers/greedy_coordinate_descent
import random


suite "Test greedy coordinate descent":
  let
    n = 50
    d = 6
    maxComponents = 6


  test "Test fitLinear":
    for refitFully in [true, false]:
      for fitIntercept in [true, false]:
        var
          X: CSCDataset
          y: seq[float64]
        createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                        explicit, false, fitIntercept)

        # fit fast version
        var cfm = newConvexFactorizationMachine(
          task = regression,  maxComponents = maxComponents,
          fitLinear = false, fitIntercept = fitIntercept)
        var gcd = newGreedyCD(maxIter = 10, verbose = 0, tol = 0,
                              refitFully=refitFully)
        gcd.fit(X, y, cfm)
        for j in 0..<d:
          check cfm.w[j] == 0.0


  test "Test fitIntercept":
    for refitFully in [true, false]:
      for fitLinear in [true, false]:
        var
          X: CSCDataset
          y: seq[float64]
        createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                        explicit, fitLinear, false)
        # fit fast version
        var cfm = newConvexFactorizationMachine(
          task = regression, maxComponents = maxComponents,
          fitLinear = fitLinear, fitIntercept = false)
        var gcd = newGreedyCD(
          maxIter = 10, verbose = 0, tol = 0, refitFully=refitFully
        )
        gcd.fit(X, y, cfm)
        check cfm.intercept == 0.0


  test "Test warmStart":
    for refitFully in [true, false]:
      for ignoreDiag in [true, false]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                            explicit, fitLinear, fitIntercept)
            randomize(1)
            var cfmWarm = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, ignoreDiag=ignoreDiag)
            var gcdWarm = newGreedyCD(
              maxIter = 1, verbose = 0, tol = 0, refitFully=refitFully,
            )
            for i in 0..<10:
              gcdWarm.fit(X, y, cfmWarm)
            randomize(1)
            var cfm = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, fitIntercept = fitIntercept,
              ignoreDiag=ignoreDiag)

            var gcd = newGreedyCD(
              maxIter = 10, verbose = 0, tol = 0, refitFully=refitFully
            )
            gcd.fit(X, y, cfm)
            check abs(cfm.intercept-cfmWarm.intercept) < 1e-5
            checkAlmostEqual(cfm.w, cfmWarm.w)
            checkAlmostEqual(cfm.lams, cfmWarm.lams)
  

  # Too slow!
  test "Comparison to naive implementation":
    for refitFully in [true, false]:
      for ignoreDiag in [true, false]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
              XMat = zeros([n, d])

            createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                            explicit, fitLinear, fitIntercept,
                            threshold=0.3)
            for j in 0..<d:
              for (i, val) in X.getCol(j):
                XMat[i, j] = val
            # fit slow version
            randomize(1)
            var cfmSlow = newCFMSlow(
              task = regression,  maxComponents = maxComponents,
              fitLinear = fitLinear, fitIntercept = fitIntercept,
              ignoreDiag = ignoreDiag)
            var gcdSlow = newGCDSlow(
              maxIter = 10, tol = 0, maxIterPower=1000, tolPower=0.0,
              refitFully=refitFully, maxIterADMM=100)
            gcdSlow.fit(XMat, y, cfmSlow)

            # fit fast version
            randomize(1)
            var cfm = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, fitIntercept = fitIntercept,
              ignoreDiag = ignoreDiag)

            var gcd = newGreedyCD(
              maxIter = 10, verbose = 0, tol = 0, tolPower=0.0,
              maxIterPower=1000, refitFully = refitFully,
              maxIterADMM=100)
            gcd.fit(X, y, cfm)

            check abs(cfm.intercept-cfmSlow.intercept) < 1e-5
            checkAlmostEqual(cfm.w, cfmSlow.w, atol=1e-5)
            checkAlmostEqual(cfm.lams, cfmSlow.lams, rtol=1e-5)
            checkAlmostEqual(cfm.P, cfm.P, atol=1e-5)


  test "Test score":
    for refitFully in [true, false]:
      for ignoreDiag in [true, false]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                            explicit, fitLinear, fitIntercept)

            var cfm = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, fitIntercept = fitIntercept,
              ignoreDiag=ignoreDiag)
            var gcd = newGreedyCD(
              maxIter = 20, verbose = 0, tol = 0, refitFully=refitFully,
              alpha0 = 1e-9, alpha = 1e-9, beta = 1e-9, 
            )
            init(cfm, X)
            let scoreBefore = cfm.score(X, y)
            gcd.fit(X, y, cfm)
            check cfm.score(X, y) < scoreBefore


  test "Test regularization":
    for refitFully in [true, false]:
      for ignoreDiag in [true, false]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                            explicit, fitLinear, fitIntercept,
                            scale=1.0)

            var cfmWeakReg = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, ignoreDiag=ignoreDiag,
              fitIntercept = fitIntercept)
            var gcdWeakReg = newGreedyCD(
              maxIter = 100, verbose = 0, tol = 0, refitFully=refitFully,
              alpha0 = 1e-5, alpha = 1e-5, beta = 1e-6
            )
            gcdWeakReg.fit(X, y, cfmWeakReg)
            
            var cfmStrongReg = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, fitIntercept = fitIntercept,
              ignoreDiag=ignoreDiag)
            var gcdStrong = newGreedyCD(
              maxIter = 100, verbose = 0, tol = 0, refitFully=refitFully,
              alpha0 = 10000000, alpha = 10000000, beta = 100000000
            )

            gcdStrong.fit(X, y, cfmStrongReg)
            check cfmWeakReg.score(X, y) < cfmStrongReg.score(X, y)
            check abs(cfmWeakReg.intercept) >= abs(cfmStrongReg.intercept)
            check norm(cfmWeakReg.w, 2) >= norm(cfmStrongReg.w, 2)
            check norm(cfmWeakReg.lams, 2) >= norm(cfmStrongReg.lams, 2)