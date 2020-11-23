import unittest
import utils, optimizer/hazan_slow, model/cfm_slow
import nimfm/dataset, nimfm/tensor/tensor
import nimfm/model/convex_factorization_machine, nimfm/model/fm_base
from nimfm/model/factorization_machine import FitLowerKind
import nimfm/optimizer/hazan
import random


suite "Test hazan":
  let
    n = 50
    d = 6
    maxComponents = 6

  test "Test fitLinear":
    for optimal in [true, false]:
      for ignoreDiag in [true, false]:
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
          var hazan = newHazan(maxIter = 10, verbose = 0, tol = 0,
                              optimal=optimal)
          hazan.fit(X, y, cfm)
          for j in 0..<d:
            check cfm.w[j] == 0.0


  test "Test fitIntercept":
    for optimal in [true, false]:
      for ignoreDiag in [true, false]:
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
          var hazan = newHazan(
            maxIter = 10, verbose = 0, tol = 0, optimal=optimal
          )
          hazan.fit(X, y, cfm)
          check cfm.intercept == 0.0


  test "Test warmStart":
    for optimal in [true, false]:
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
            var hazanWarm = newHazan(
              maxIter = 1, verbose = 0, tol = -100, optimal=optimal,
            )
            for i in 0..<100:
              hazanWarm.fit(X, y, cfmWarm)

            randomize(1)
            var cfm = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, fitIntercept = fitIntercept,
              ignoreDiag=ignoreDiag)

            var hazan = newHazan(
              maxIter = 100, verbose = 0, tol = -100, optimal=optimal
            )
            hazan.fit(X, y, cfm)
            check abs(cfm.intercept-cfmWarm.intercept) < 1e-5
            checkAlmostEqual(cfm.w, cfmWarm.w)
            checkAlmostEqual(cfm.lams, cfmWarm.lams)

  # Too slow!
  test "Comparison to naive implementation":
    for optimal in [true, false]:
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
            var hazanSlow = newHazanSlow(
              maxIter = 50, tol = 0, maxIterPower=1000, tolPower=0.0,
              optimal=optimal)
            hazanSlow.fit(XMat, y, cfmSlow)
            # fit fast version
            randomize(1)
            var cfm = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, fitIntercept = fitIntercept,
              ignoreDiag = ignoreDiag)

            var hazan = newHazan(
              maxIter = 50, verbose = 0, tol = 0, tolPower=0.0,
              maxIterPower=1000, optimal = optimal)
            hazan.fit(X, y, cfm)
            
            check abs(cfm.intercept-cfmSlow.intercept) < 1e-5
            checkAlmostEqual(cfm.w, cfmSlow.w)
            checkAlmostEqual(cfm.lams, cfmSlow.lams, atol=1e-7)
            checkAlmostEqual(cfm.P, cfmSlow.P)


  test "Test score":
    for optimal in [true, false]:
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
              ignoreDiag = ignoreDiag)

            var hazan = newHazan(
              maxIter = 100, verbose = 0, tol = 0, tolPower=0.0,
              maxIterPower=1000, optimal = optimal, eta=0.1)

            init(cfm, X)
            let scoreBefore = cfm.score(X, y)
            hazan.fit(X, y, cfm)
            check cfm.score(X, y) < scoreBefore + 1e-9


  test "Test regularization":
    for optimal in [true, false]:
      for ignoreDiag in [true, false]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, 2, maxComponents div 2, 42,
                            explicit, fitLinear, fitIntercept,
                            scale=3.0)

            var cfmWeakReg = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, warmStart = true, ignoreDiag=ignoreDiag,
              fitIntercept = fitIntercept)
            var hazan = newHazan(
              maxIter = 100, verbose = 0, tol = -100, optimal=optimal, eta=1000)
            hazan.fit(X, y, cfmWeakReg)
            
            var cfmStrongReg = newConvexFactorizationMachine(
              task = regression, maxComponents = maxComponents,
              fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, ignoreDiag=ignoreDiag)
            var hazanStrong = newHazan(
              maxIter = 100, verbose = 0, tol = -100, optimal=optimal, eta=0.01)
            hazanStrong.fit(X, y, cfmStrongReg)

            check norm(cfmWeakReg.lams, 1) >= norm(cfmStrongReg.lams, 1)
