import unittest
import utils, optimizers/pbcd_slow
import nimfm/loss, nimfm/dataset, nimfm/tensor/tensor
import nimfm/models/factorization_machine, nimfm/models/fm_base
import nimfm/optimizers/pbcd
import nimfm/regularizers/regularizers
import regularizers/l1_slow
import models/fm_slow


suite "Test proximal block coordinate descent for L1":
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
            reg = newL1()
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, false, fitIntercept)

          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = false,
            fitIntercept = fitIntercept, randomState = 1)
          var pbcd = newPBCD(maxIter = 10, verbose = 0, tol = 0, reg=reg)
          pbcd.fit(X, y, fm)
          for j in 0..<d:
            check fm.w[j] == 0.0


  test "Test fitIntercept":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          var
            X: CSCDataset
            y: seq[float64]
            reg = newL1()
          createFMDataset(X, y, n, d, degree, nComponents, 42,
                          fitLower, fitLinear, false)
          # fit fast version
          var fm = newFactorizationMachine(
            task = regression, degree = degree, nComponents = nComponents,
            fitLower = fitLower, fitLinear = fitLinear,
            fitIntercept = false, randomState = 1)
          var pbcd = newPBCD(
            maxIter = 10, verbose = 0, tol = 0, reg=reg
          )
          pbcd.fit(X, y, fm)
          check fm.intercept == 0.0

  test "Test warmStart":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            for maxSearch in [0, 20]:
              var
                X: CSCDataset
                y: seq[float64]
                reg = newL1()

              createFMDataset(X, y, n, d, degree, nComponents, 42,
                              fitLower, fitLinear, fitIntercept)
              var fmWarm = newFactorizationMachine(
                task = regression, degree = degree, nComponents = nComponents,
                fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
                fitIntercept = fitIntercept, randomState = 1)
              var pbcdWarm = newPBCD(
                maxIter = 1, verbose = 0, tol = 0, reg=reg, maxSearch = maxSearch
              )
              for i in 0..<10:
                pbcdWarm.fit(X, y, fmWarm)

              var fm = newFactorizationMachine(
                task = regression, degree = degree, nComponents = nComponents,
                fitLower = fitLower, fitLinear = fitLinear,
                fitIntercept = fitIntercept, randomState = 1)

              var pbcd = newPBCD(
                maxIter = 10, verbose = 0, tol = 0, reg=reg, maxSearch = maxSearch
              )
              pbcd.fit(X, y, fm)

              check abs(fm.intercept-fmWarm.intercept) < 1e-8
              checkAlmostEqual(fm.w, fmWarm.w, atol = 1e-8)
              checkAlmostEqual(fm.P, fmWarm.P, atol = 1e-8)
  
  # Too slow!
  test "Comparison to naive implementation":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            for maxSearch in [0, 20]:
              var
                X: CSCDataset
                y: seq[float64]
                XMat = zeros([n, d])
                reg = newL1()
                regSlow = newL1Slow()
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
              var pbcdSlow = newPBCDSlow(maxIter = 3, tol = 0, reg=regSlow,
                                         maxSearch = maxSearch, gamma=1e-5, beta=1e-5)
              pbcdSlow.fit(XMat, y, fmSlow)
              # fit fast version
              var fm = newFactorizationMachine(
                task = regression, degree = degree, nComponents = nComponents,
                fitLower = fitLower, fitLinear = fitLinear,
                fitIntercept = fitIntercept, randomState = 1)
              var pbcd = newPBCD(
                maxIter = 3, verbose = 0, tol = 0, reg=reg, maxSearch=maxSearch,
                gamma = 1e-5, beta = 1e-5
              )
              pbcd.fit(X, y, fm)
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
              reg = newL1()

            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept)

            var fm = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear,
              fitIntercept = fitIntercept, randomState = 1)
            var pbcd = newPBCD(
              maxIter = 20, verbose = 0, tol = 0, alpha0 = 1e-9, alpha = 1e-9,
              beta = 1e-9, gamma=1e-9, reg=reg
            )
            fm.init(X)
            let scoreBefore = fm.score(X, y)
            pbcd.fit(X, y, fm)
            check fm.score(X, y) < scoreBefore


  test "Test regularization":
    for degree in 2..<5:
      for fitLower in [explicit, none, augment]:
        for fitLinear in [true, false]:
          for fitIntercept in [true, false]:
            var
              X: CSCDataset
              y: seq[float64]
              reg = newL1()

            createFMDataset(X, y, n, d, degree, nComponents, 42,
                            fitLower, fitLinear, fitIntercept, scale=3.0)

            var fmWeakReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1)
            var pbcd = newPBCD(
              maxIter = 100, verbose = 0, tol = 0,
              alpha0=0, alpha=0, beta=0, gamma=0, reg=reg
            )
            pbcd.fit(X, y, fmWeakReg)
            
            var fmStrongReg = newFactorizationMachine(
              task = regression, degree = degree, nComponents = nComponents,
              fitLower = fitLower, fitLinear = fitLinear, warmStart = true,
              fitIntercept = fitIntercept, randomState = 1)
            var pbcdStrong = newPBCD(
              maxIter = 100, verbose = 0, tol = 0,
              alpha0=100000, alpha=100000, beta=100000, gamma=100000, reg=reg
            )
            pbcdStrong.fit(X, y, fmStrongReg)
        
            check fmWeakReg.score(X, y) < fmStrongReg.score(X, y)
            check abs(fmWeakReg.intercept) >= abs(fmStrongReg.intercept)
            check norm(fmWeakReg.w, 2) >= norm(fmStrongReg.w, 2)
            check norm(fmWeakReg.P, 2) >= norm(fmStrongReg.P, 2)
