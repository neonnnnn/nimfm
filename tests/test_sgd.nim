import unittest
import utils, optimizers/sgd_slow
import nimfm/loss, nimfm/dataset, nimfm/tensor/tensor
import nimfm/models/factorization_machine, nimfm/models/fm_base
import nimfm/optimizers/sgd
import models/fm_slow


suite "Test stochastic gradient descent":
  let
    n = 80
    d = 8
    nComponents = 4


  test "Test fitLinear":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitIntercept in [true, false]:
          var
            X: CSRDataset
            y: seq[float64]
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, false, fitIntercept)

          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = false,
            fitIntercept = fitIntercept, randomState = 1)
          var sgd = newSGD(maxIter = 10, verbose = 0, tol = 0)
          sgd.fit(X, y, fm)
          for j in 0..<d:
            check fm.w[j] == 0.0


  test "Test fitIntercept":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          var
            X: CSRDataset
            y: seq[float64]
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, fitLinear, false)
          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = fitLinear,
            fitIntercept = false, randomState = 1)
          var sgd = newSGD(
            maxIter = 10, verbose = 0, tol = 0
          )
          sgd.fit(X, y, fm)
          check fm.intercept == 0.0


  test "Test warmStart":
    for degree in 2..<5:
      for fitLower in [explicit, augment, none]:
        for fitLinear in [false, true]:
          for fitIntercept in [false, true]:
            var
              X: CSRDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)
            var fmWarm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1)
            var sgdWarm = newSGD(
              maxIter = 1, verbose = 0, tol = 0, shuffle = false
            )
            for i in 0..<10:
              sgdWarm.fit(X, y, fmWarm)
            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, 
              fitIntercept = fitIntercept, randomState = 1)

            var sgd = newSGD(
              maxIter = 10, verbose = 0, tol = 0, shuffle = false
            )
            sgd.fit(X, y, fm)

            check abs(fm.intercept-fmWarm.intercept) < 1e-8
            checkAlmostEqual(fm.w, fmWarm.w, atol = 1e-8)
            checkAlmostEqual(fm.P, fmWarm.P, atol = 1e-8)


  test "Comparison to naive implementation":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [false, true]:
          for fitIntercept in [false, true]:
            var
              X: CSRDataset
              y: seq[float64]
              XMat = zeros([n, d])
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept,
                            threshold=0.3)
            for i in 0..<n:
              for (j, val) in X.getRow(i):
                XMat[i, j] = val
            # fit slow version
            var fmSlow = newFMSlow(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var sgdSlow = newSGDSlow(maxIter = 5, tol = 0)
            sgdSlow.fit(XMat, y, fmSlow)

            # fit fast version
            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var sgd = newSGD(
              maxIter = 5, verbose = 0, tol = 0
            )
            sgd.fit(X, y, fm)
            check abs(fm.intercept-fmSlow.intercept) < 1e-7
            checkAlmostEqual(fm.w, fmSlow.w)
            checkAlmostEqual(fm.P, fmSlow.P)


  test "Test score":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSRDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)

            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var sgd = newSGD(
              maxIter = 20, verbose = 0, tol = 0,
              alpha0 = 1e-9, alpha = 1e-9, beta = 1e-9,
            )
            fm.init(X)
            let scoreBefore = fm.score(X, y)
            sgd.fit(X, y, fm)
            check fm.score(X, y) < scoreBefore


  test "Test regularization":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSRDataset
              y: seq[float64]
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept,
                            scale=1.0)

            var fmWeakReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = false,
              fitIntercept = fitIntercept, randomState = 1)
            var sgd = newSGD(
              maxIter = 200, verbose = 0, tol = 0,
              alpha0 = 1e-9, alpha = 1e-9, beta = 1e-9,
            )
            sgd.fit(X, y, fmWeakReg)
            
            var fmStrongReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = false,
              fitIntercept = fitIntercept, randomState = 1)
            var sgdStrong = newSGD(
              maxIter = 200, verbose = 0, tol = 0,
              alpha0 = 1000000, alpha = 1000000, beta = 1000000,
            )
            sgdStrong.fit(X, y, fmStrongReg)

            var normWeak = norm(fmWeakReg.P, 2) + norm(fmWeakReg.w, 2)
            normWeak += fmWeakReg.intercept * fmWeakReg.intercept
            var normStrong = norm(fmStrongReg.P, 2) + norm(fmStrongReg.w, 2)
            normStrong = fmStrongReg.intercept * fmStrongReg.intercept
            check normWeak >= normStrong