import unittest
import utils, optimizer/sgd_ffm_slow
import nimfm/loss, nimfm/dataset, nimfm/tensor/tensor
import nimfm/model/field_aware_factorization_machine, nimfm/model/fm_base
import nimfm/optimizer/sgd_ffm
import model/ffm_slow


suite "Test stochastic gradient descent for FFM":
  let
    n = 80
    d = 20
    nFields = 5
    nComponents = 4

  test "Test fitLinear":
    for fitIntercept in [true, false]:
      var
        X: CSRFieldDataset
        y: seq[float64]
        fieldDict: seq[int]
      createFFMDataset(X, y, fieldDIct, n, d, nFields, nComponents, 42,
                       false, fitIntercept)
      # fit fast version
      var ffm = newFieldAwareFactorizationMachine(
        task = regression, nComponents = nComponents,
        fitLinear = false,
        fitIntercept = fitIntercept, randomState = 1)
      var sgd = newSGD(maxIter = 10, verbose = 0, tol = 0)
      sgd.fit(X, y, ffm)
      for j in 0..<d:
        check ffm.w[j] == 0.0


  test "Test fitIntercept":
      for fitLinear in [true, false]:
        var
          X: CSRFieldDataset
          y: seq[float64]
          fieldDict: seq[int]
        createFFMDataset(X, y, fieldDict, n, d, nFields, nComponents, 42,
                         fitLinear, false)
        # fit fast version
        var ffm = newFieldAwareFactorizationMachine(
          task = regression, nComponents = nComponents,
          fitLinear = fitLinear,
          fitIntercept = false, randomState = 1)
        var sgd = newSGD(
          maxIter = 10, verbose = 0, tol = 0
        )
        sgd.fit(X, y, ffm)
        check ffm.intercept == 0.0


  test "Test warmStart":
    for fitLinear in [false, true]:
      for fitIntercept in [false, true]:
        var
          X: CSRFieldDataset
          y: seq[float64]
          fieldDict: seq[int]
        createFFMDataset(X, y, fieldDict, n, d, nFields,  nComponents, 42,
                         fitLinear, fitIntercept)
        var ffmWarm = newFieldAwareFactorizationMachine(
          task = regression, nComponents = nComponents,
          fitLinear = fitLinear, warmStart = true,
          fitIntercept = fitIntercept, randomState = 1)
        var sgdWarm = newSGD(
          maxIter = 1, verbose = 0, tol = 0, shuffle = false
        )
        for i in 0..<10:
          sgdWarm.fit(X, y, ffmWarm)
        var ffm = newFieldAwareFactorizationMachine(
          task = regression, nComponents = nComponents,
          fitLinear = fitLinear, 
          fitIntercept = fitIntercept, randomState = 1)

        var sgd = newSGD(
          maxIter = 10, verbose = 0, tol = 0, shuffle = false
        )
        sgd.fit(X, y, ffm)

        check abs(ffm.intercept-ffmWarm.intercept) < 1e-8
        checkAlmostEqual(ffm.w, ffmWarm.w, atol = 1e-8)
        checkAlmostEqual(ffm.P, ffmWarm.P, atol = 1e-8)

  test "Comparison to naive implementation":
    for fitLinear in [false, true]:
      for fitIntercept in [false, true]:
        var
          X: CSRFieldDataset
          y: seq[float64]
          XMat = zeros([n, d])
          fieldDict: seq[int]
        createFFMDataset(X, y, fieldDict, n, d, nFields, nComponents, 42,
                         fitLinear, fitIntercept, threshold=0.3)
        for i in 0..<n:
          for (j, val) in X.getRow(i):
            XMat[i, j] = val
        # fit slow version
        var ffmSlow = newFFMSlow(
          task = regression, nComponents = nComponents, fitLinear = fitLinear,
          fitIntercept = fitIntercept, randomState = 1)
        var sgdSlow = newSGDSlow(maxIter = 5, tol = 0, shuffle=false)
        sgdSlow.fit(XMat, fieldDict, y, ffmSlow)
        
        # fit fast version
        var ffm = newFieldAwareFactorizationMachine(
          task = regression, nComponents = nComponents, fitLinear = fitLinear,
          fitIntercept = fitIntercept, randomState = 1)
        var sgd = newSGD( maxIter = 5, verbose = 0, tol = 0, shuffle=false)
        sgd.fit(X, y, ffm)
        check abs(ffm.intercept-ffmSlow.intercept) < 1e-7
        checkAlmostEqual(ffm.w, ffmSlow.w)
        checkAlmostEqual(ffm.P, ffmSlow.P)

  test "Test score":
    for fitLinear in [true, false]:
      for fitIntercept in [true, false]:
        var
          X: CSRFieldDataset
          y: seq[float64]
          fieldDict: seq[int]
        createFFMDataset(X, y, fieldDict, n, d, nFields, nComponents, 42,
                         fitLinear, fitIntercept)

        var ffm = newFieldAwareFactorizationMachine(
          task = regression, nComponents = nComponents, fitLinear = fitLinear,
          fitIntercept = fitIntercept, randomState = 1)
        var sgd = newSGD(
          maxIter = 20, verbose = 0, tol = 0,
          alpha0 = 1e-9, alpha = 1e-9, beta = 1e-9,
        )
        ffm.init(X)
        let scoreBefore = ffm.score(X, y)
        sgd.fit(X, y, ffm)
        check ffm.score(X, y) < scoreBefore


  test "Test regularization":
    for fitLinear in [true, false]:
      for fitIntercept in [true, false]:
        var
          X: CSRFieldDataset
          y: seq[float64]
          fieldDict: seq[int]
        createFFMDataset(X, y, fieldDict, n, d, nFields, nComponents, 42,
                         fitLinear, fitIntercept, scale=1.0)

        var ffmWeakReg = newFieldAwareFactorizationMachine(
          task = regression, nComponents = nComponents, fitLinear = fitLinear,
          warmStart = false, fitIntercept = fitIntercept, randomState = 1)
        var sgd = newSGD(
          maxIter = 200, verbose = 0, tol = 0,
          alpha0 = 1e-9, alpha = 1e-9, beta = 1e-9,
        )
        sgd.fit(X, y, ffmWeakReg)
            
        var ffmStrongReg = newFieldAwareFactorizationMachine(
          task = regression, nComponents = nComponents, fitLinear = fitLinear,
          warmStart = false, fitIntercept = fitIntercept, randomState = 1)
        var sgdStrong = newSGD(
          maxIter = 200, verbose = 0, tol = 0,
          alpha0 = 1000000, alpha = 1000000, beta = 1000000,
        )
        sgdStrong.fit(X, y, ffmStrongReg)

        var normWeak = norm(ffmWeakReg.P, 2) + norm(ffmWeakReg.w, 2)
        normWeak += ffmWeakReg.intercept * ffmWeakReg.intercept
        var normStrong = norm(ffmStrongReg.P, 2) + norm(ffmStrongReg.w, 2)
        normStrong = ffmStrongReg.intercept * ffmStrongReg.intercept
        check normWeak >= normStrong