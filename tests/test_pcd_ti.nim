import unittest
import utils, optimizers/pcd_slow
import nimfm/loss, nimfm/dataset, nimfm/tensor/tensor
import nimfm/models/factorization_machine, nimfm/models/fm_base
import nimfm/optimizers/pcd
import nimfm/regularizers/regularizers
import regularizers/omegati_slow
import models/fm_slow


suite "Test proximal coordinate descent for OmegaTI":
  let
    n = 50
    d = 6
    nComponents = 4

  test "Test fitLinear":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitIntercept in [true, false]:
          var
            X: CSCDataset
            y: seq[float64]
            reg = newOmegaTI()
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, false, fitIntercept)

          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = false,
            fitIntercept = fitIntercept, randomState = 1)
          var pcd = newPCD(maxIter = 10, verbose = 0, tol = 0, reg=reg)
          pcd.fit(X, y, fm)
          for j in 0..<d:
            check fm.w[j] == 0.0


  test "Test fitIntercept":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          var
            X: CSCDataset
            y: seq[float64]
            reg = newOmegaTI()
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, fitLinear, false)
          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = fitLinear,
            fitIntercept = false, randomState = 1)
          var pcd = newPCD(
            maxIter = 10, verbose = 0, tol = 0, reg=reg
          )
          pcd.fit(X, y, fm)
          check fm.intercept == 0.0


  test "Test warmStart":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
              reg = newOmegaTI()

            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)
            var fmWarm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1)
            var pcdWarm = newPCD(
              maxIter = 1, verbose = 0, tol = 0, reg=reg
            )
            for i in 0..<10:
              pcdWarm.fit(X, y, fmWarm)

            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)

            var pcd = newPCD(
              maxIter = 10, verbose = 0, tol = 0, reg=reg
            )
            pcd.fit(X, y, fm)

            check abs(fm.intercept-fmWarm.intercept) < 1e-8
            checkAlmostEqual(fm.w, fmWarm.w, atol = 1e-8)
            checkAlmostEqual(fm.P, fmWarm.P, atol = 1e-8)

  # Too slow!
  test "Comparison to naive implementation":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
              XMat = zeros([n, d])
              reg = newOmegaTI()
              regSlow = newOmegaTISlow()
            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept,
                            threshold=0.3)
            for j in 0..<d:
              for (i, val) in X.getCol(j):
                XMat[i, j] = val
            # fit slow version
            var fmSlow = newFMSlow(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var pcdSlow = newPCDSlow(maxIter = 3, tol = 0, reg=regSlow)
            pcdSlow.fit(XMat, y, fmSlow)

            # fit fast version
            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var pcd = newPCD(
              maxIter = 3, verbose = 0, tol = 0, reg=reg
            )
            pcd.fit(X, y, fm)

            check abs(fm.intercept-fmSlow.intercept) < 1e-7
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
              reg = newOmegaTI()

            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)

            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var pcd = newPCD(
              maxIter = 20, verbose = 0, tol = 0,
              alpha0 = 1e-9, alpha = 1e-9, beta = 1e-9,
              gamma=1e-9, reg=reg
            )
            fm.init(X)
            let scoreBefore = fm.score(X, y)
            pcd.fit(X, y, fm)
            check fm.score(X, y) < scoreBefore


  test "Test regularization":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
              reg = newOmegaTI()

            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept,
                            scale=3.0)

            var fmWeakReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1)
            var pcd = newPCD(
              maxIter = 100, verbose = 0, tol = 0,
              alpha0=0, alpha=0, beta=0, gamma=0, reg=reg
            )
            pcd.fit(X, y, fmWeakReg)
            
            var fmStrongReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1)
            var pcdStrong = newPCD(
              maxIter = 100, verbose = 0, tol = 0,
              alpha0=100000, alpha=100000, beta=100000, gamma=100000, reg=reg
            )
            pcdStrong.fit(X, y, fmStrongReg)

            check fmWeakReg.score(X, y) < fmStrongReg.score(X, y)
            check abs(fmWeakReg.intercept) >= abs(fmStrongReg.intercept)
            check norm(fmWeakReg.w, 2) >= norm(fmStrongReg.w, 2)
            check norm(fmWeakReg.P, 2) >= norm(fmStrongReg.P, 2)