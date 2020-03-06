import unittest
import utils, cd_slow
import nimfm/loss, nimfm/dataset, nimfm/tensor, nimfm/factorization_machine
import nimfm/optimizers/coordinate_descent


suite "Test coordinate descent":
  let
    n = 100
    d = 10
    nComponents = 4

  test "Test fitLinear":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitIntercept in [true, false]:
          var
            X: CSCDataset
            y: seq[float64]
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, false, fitIntercept)
          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = false,
            fitIntercept = fitIntercept, randomState = 1)
          var cd = newCoordinateDescent(maxIter = 10, verbose = false, tol = 0)
          cd.fit(X, y, fm)
          for j in 0..<nComponents:
            check fm.w[j] == 0.0

  test "Test fitIntercept":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          var
            X: CSCDataset
            y: seq[float64]
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, fitLinear, false)
          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = fitLinear,
            fitIntercept = false, randomState = 1)
          var cd = newCoordinateDescent(
            maxIter = 10, verbose = false, tol = 0
          )
          cd.fit(X, y, fm)
          check fm.intercept == 0.0

  test "Test warmStart":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)
            # fit slow version
            var fmWarm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1)
            var cdWarm = newCoordinateDescent(
              maxIter = 1, verbose = false, tol = 0
            )
            for i in 0..<10:
              cdWarm.fit(X, y, fmWarm)

            # fit fast version
            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var cd = newCoordinateDescent(
              maxIter = 10, verbose = false, tol = 0
            )
            cd.fit(X, y, fm)

            check abs(fm.intercept-fmWarm.intercept) < 1e-8
            checkAlmostEqual(fm.w, fmWarm.w, atol = 1e-8)
            checkAlmostEqual(fm.P, fmWarm.P, atol = 1e-8)

  test "Comarpasion to naive implementation":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)
            # fit slow version
            var fmSlow = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var cdSlow = newCoordinateDescentSlow(maxIter = 10, tol = 0)
            cdSlow.fit(X, y, fmSlow)

            # fit fast version
            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var cd = newCoordinateDescent(
              maxIter = 10, verbose = false, tol = 0
            )
            cd.fit(X, y, fm)

            check abs(fm.intercept-fmSlow.intercept) < 1e-3
            checkAlmostEqual(fm.w, fmSlow.w)
            checkAlmostEqual(fm.P, fmSlow.P)


  test "Test score":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)

            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, alpha0 = 1e-9,
              alpha = 1e-9, beta = 1e-9,
              fitIntercept = fitIntercept, randomState = 1)
            var cd = newCoordinateDescent(
              maxIter = 20, verbose = false, tol = 0
            )
            fm.init(X)
            let scoreBefore = fm.score(X, y)
            cd.fit(X, y, fm)
            check fm.score(X, y) < scoreBefore


  test "Test regularization":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)

            var fmWeakReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1,
              alpha0 = 1e-9, alpha = 1e-9, beta = 1e-9)
            var cd = newCoordinateDescent(
              maxIter = 10, verbose = false, tol = 0
            )
            cd.fit(X, y, fmWeakReg)

            var fmStrongReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1,
              alpha0 = 100, alpha = 100, beta = 100)
            cd.fit(X, y, fmStrongReg)

            check fmWeakReg.score(X, y) < fmStrongReg.score(X, y)
            check abs(fmWeakReg.intercept) >= abs(fmStrongReg.intercept)
            check norm(fmWeakReg.w, 2) >= norm(fmStrongReg.w, 2)
            check norm(fmWeakReg.P, 2) >= norm(fmStrongReg.P, 2)
