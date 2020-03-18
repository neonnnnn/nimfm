import unittest
import utils, greedy_cd_slow, cfm_slow
import nimfm/loss, nimfm/dataset, nimfm/tensor
import nimfm/convex_factorization_machine, nimfm/fm_base
from nimfm/factorization_machine import FitLowerKind
import nimfm/optimizers/greedy_coordinate_descent
import random


suite "Test greedy coordinate descent":
  let
    n = 50
    d = 6
    maxComponents = 6


  test "Test fitLinear":
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
      var gcd = newGreedyCoordinateDescent(maxIter = 10, verbose = 0, tol = 0)
      gcd.fit(X, y, cfm)
      for j in 0..<d:
        check cfm.w[j] == 0.0


  test "Test fitIntercept":
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
      var gcd = newGreedyCoordinateDescent(
        maxIter = 10, verbose = 0, tol = 0
      )
      gcd.fit(X, y, cfm)
      check cfm.intercept == 0.0


  test "Test warmStart":
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
          fitIntercept = fitIntercept)
        var gcdWarm = newGreedyCoordinateDescent(
          maxIter = 1, verbose = 0, tol = 0
        )
        for i in 0..<10:
          gcdWarm.fit(X, y, cfmWarm)

        randomize(1)
        var cfm = newConvexFactorizationMachine(
          task = regression, maxComponents = maxComponents,
          fitLinear = fitLinear, fitIntercept = fitIntercept)

        var gcd = newGreedyCoordinateDescent(
          maxIter = 10, verbose = 0, tol = 0
        )
        gcd.fit(X, y, cfm)
        check abs(cfm.intercept-cfmWarm.intercept) < 1e-5
        checkAlmostEqual(cfm.w, cfmWarm.w, atol = 1e-5)
        checkAlmostEqual(cfm.lams, cfmWarm.lams, atol = 1e-5)
  

  # Too slow!
  test "Comparison to naive implementation":
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
          fitLinear = fitLinear, fitIntercept = fitIntercept)
        var gcdSlow = newGCDSlow(
          maxIter = 10, tol = 0, maxIterPower=1000, tolPower=0.0)
        gcdSlow.fit(XMat, y, cfmSlow)

        # fit fast version
        randomize(1)
        var cfm = newConvexFactorizationMachine(
          task = regression, maxComponents = maxComponents,
          fitLinear = fitLinear, fitIntercept = fitIntercept)
        var gcd = newGreedyCoordinateDescent(
          maxIter = 10, verbose = 0, tol = 0, tolPower=0.0,
          maxIterPower=1000
        )
        gcd.fit(X, y, cfm)

        #check cfm.score(X, y) == cfmSlow.score(X, y)
        check abs(cfm.intercept-cfmSlow.intercept) < 1e-3
        checkAlmostEqual(cfm.w, cfmSlow.w)
        checkAlmostEqual(cfm.lams, cfmSlow.lams)
        
  test "Test score":
    for fitLinear in [true, false]:
      for fitIntercept in [true, false]:
        var
          X: CSCDataset
          y: seq[float64]
        createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                        explicit, fitLinear, fitIntercept)

        var cfm: ConvexFactorizationMachine = newConvexFactorizationMachine(
          task = regression, maxComponents = maxComponents,
          fitLinear = fitLinear, alpha0 = 1e-9,
          alpha = 1e-9, beta = 1e-9, fitIntercept = fitIntercept)
        var gcd = newGreedyCoordinateDescent(
          maxIter = 20, verbose = 0, tol = 0
        )
        init(cfm, X)
        let scoreBefore = cfm.score(X, y)
        gcd.fit(X, y, cfm)
        check cfm.score(X, y) < scoreBefore


  test "Test regularization":
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
          fitLinear = fitLinear, warmStart = true,
          fitIntercept = fitIntercept, alpha0 = 0, alpha = 0, beta = 0)
        var gcd = newGreedyCoordinateDescent(
          maxIter = 100, verbose = 0, tol = 0
        )
        gcd.fit(X, y, cfmWeakReg)
            
        var cfmStrongReg = newConvexFactorizationMachine(
          task = regression, maxComponents = maxComponents,
          fitLinear = fitLinear, warmStart = true,
          fitIntercept = fitIntercept,
          alpha0 = 1000, alpha = 1000, beta = 1000)
        gcd.fit(X, y, cfmStrongReg)

        check cfmWeakReg.score(X, y) < cfmStrongReg.score(X, y)
        check abs(cfmWeakReg.intercept) >= abs(cfmStrongReg.intercept)
        check norm(cfmWeakReg.w, 2) >= norm(cfmStrongReg.w, 2)
        check norm(cfmWeakReg.lams, 2) >= norm(cfmStrongReg.lams, 2)