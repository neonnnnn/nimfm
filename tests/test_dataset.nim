import unittest
import nimfm/dataset, nimfm/tensor/sparse
import os, sequtils, sugar, math
from utils import createFMDataset


proc checkDenseCSR[T](dense: seq[seq[float64]], sparse: T) =
  let nSamples = sparse.nSamples
  let nFeatures = sparse.nFeatures
  check nSamples == len(dense)
  for i in 0..<nSamples:
    check nFeatures == len(dense[i])
  
  var nnzDense = 0
  var nnzSparse = 0
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      if dense[i][j] != 0:
        nnzDense += 1

  for i in 0..<nSamples:
    for (j, val) in sparse.getRow(i):
      check abs(dense[i][j] - val) < 1e-10
      nnzSparse += 1

  check nnzDense == nnzSparse
  check nnzDense == sparse.nnz


proc checkDenseCSC[T](dense: seq[seq[float64]], sparse: T) =
  let nSamples = sparse.nSamples
  let nFeatures = sparse.nFeatures
  check nSamples == len(dense)
  for i in 0..<nSamples:
    check nFeatures == len(dense[i])
  
  var nnzDense = 0
  var nnzSparse = 0
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      if dense[i][j] != 0.0:
        nnzDense += 1

  for j in 0..<nFeatures:
    for (i, val) in sparse.getCol(j):
      check abs(dense[i][j] - val) < 1e-10
      nnzSparse += 1
  
  check nnzDense == nnzSparse
  check nnzDense == sparse.nnz


proc normalizeL1(X: var seq[seq[float64]], axis=1) =
  if axis == 1:
    for i in 0..<len(X):
      let norm =  sum(X[i].map(x=>abs(x)))
      if norm != 0:
        X[i].apply(x=>x/norm)
  elif axis == 0:
    for j in 0..<len(X[0]):
      var norm = 0.0
      for i in 0..<len(X):
        norm += abs(X[i][j])
      if norm != 0.0:
        for i in 0..<len(X):
          X[i][j] /= norm
        

proc normalizeL2(X: var seq[seq[float64]], axis=1) =
  if axis == 1:
    for i in 0..<len(X):
      let norm =  sqrt(sum(X[i].map(x=>x^2)))
      if norm != 0:
        X[i].apply(x=>x/norm)
  elif axis == 0:
    for j in 0..<len(X[0]):
      var norm = 0.0
      for i in 0..<len(X):
        norm += X[i][j]^2
      norm = sqrt(norm)
      if norm != 0.0:
        for i in 0..<len(X):
          X[i][j] /= norm


proc normalizeLinfty(X: var seq[seq[float64]], axis=1) =
  if axis == 1:
    for i in 0..<len(X):
      let norm =  max(X[i].map(x=>abs(x)))
      if norm != 0:
        X[i].apply(x=>x/norm)
  elif axis == 0:
    for j in 0..<len(X[0]):
      var norm = 0.0
      for i in 0..<len(X):
        norm = max(abs(X[i][j]), norm)
      if norm != 0.0:
        for i in 0..<len(X):
          X[i][j] /= norm


proc vstack(dataseq: seq[seq[seq[float64]]]): seq[seq[float64]] =
  result = @[]
  for data in dataseq:
    for row in data:
      result.add(row)
      if result[0].len != result[^1].len:
        raise newException(ValueError, "Different nFeatures.")


proc hstack(dataseq: seq[seq[seq[float64]]]): seq[seq[float64]] =
  result = dataseq[0]
  for data in dataseq[1..^1]:
    if len(result) != len(data):
      raise newException(ValueError, "Different nSamples.")
    for i in 0..<len(data):
      result[i].add(data[i])


suite "Test datasets":
  const nSamples: int = 4
  var data: seq[seq[float64]]
  newSeq(data, 4)
  data[0] = @[0.0, 0.0, 1.0, 0.0, -4.2, 0.0]
  data[1] = @[-3.0, 0.0, 0, 0, 0.0, 0.0]
  data[2] = @[0.0, 0.0, 0, 0, 0, 0]
  data[3] = @[0.0, 0.0, 0, -5.0, 0.0, 103.2]
  let yTrue: array[nSamples, float] = [-1.0, 2.0, -10, 5.2]

  # create file
  dumpSVMLightFile("testsample.svm", data, toSeq(yTrue))
  convertSVMLightFile("testsample.svm", "testsample", "testlabel")
  transposeFile("testsample", "testsampleT")
  transposeFile("testsampleT", "testsampleTT")

  echo("create a large dataset")
  const nSamplesLarge = 2000
  var datasetLarge: CSRDataset
  var yTrueLarge: seq[float64]
  createFMDataset(datasetLarge, yTrueLarge, nSamplesLarge, 100, 2, 1, 1, threshold=0.3)
  var dataLarge = datasetLarge.toSeq()

  dumpSVMLightFile("testsample_large.svm", datasetLarge, yTrueLarge)
  convertSVMLightFile("testsample_large.svm", "testsample_large", "testlabel_large")
  transposeFile("testsample_large", "testsample_largeT")
  transposeFile("testsample_largeT", "testsample_largeTT")


  test "Test CSRDataset":
    var dataset: CSRDataset
    var y: seq[float]
    loadSVMLightFile("testsample.svm", dataset, y)
    checkDenseCSR(data, dataset)

    dumpSVMLightFile("testsample_dumped.svm", dataset, y)
    var dataset2: CSRDataset
    var y2: seq[float]
    loadSVMLightFile("testsample_dumped.svm", dataset2, y2)
    checkDenseCSR(data, dataset2)


  test "Test toCSR":
    var dataset = toCSRDataset(data)
    checkDenseCSR(data, dataset)
    var datasetCSC: CSCDataset
    var y: seq[float]
    loadSVMLightFile("testsample.svm", datasetCSC, y)
    checkDenseCSR(data, toCSRDataset(datasetCSC))


  test "Test streamLabel":
    var yStream = loadStreamLabel("testlabel")
    for i in 0..<nSamples:
      check abs(yStream[i] - yTrue[i]) < 1e-10
  

  test "Test streamLabelLarge":
    var yStream = loadStreamLabel("testlabel_large")
    for i in 0..<nSamplesLarge:
      check abs(yStream[i] - yTrueLarge[i]) < 1e-10


  test "Test load/dump StreamCSR":
    var dataset = toCSRDataset(data)
    var datasetStream = newStreamCSRDataset("testsample")
    check dataset.shape == datasetStream.shape
    check dataset.nnz == datasetStream.nnz
    checkDenseCSR(data, datasetStream)


  test "Test load/dump StreamCSC":
    var dataset = toCSCDataset(data)
    var datasetStream = newStreamCSCDataset("testsampleT")
    check dataset.shape == datasetStream.shape
    check dataset.nnz == datasetStream.nnz
    checkDenseCSC(data, datasetStream)


  test "Test load/dump StreamCSR2":
    var dataset = toCSRDataset(data)
    var datasetStream = newStreamCSRDataset("testsampleTT")
    check dataset.shape == datasetStream.shape
    check dataset.nnz == datasetStream.nnz
    checkDenseCSR(data, datasetStream)


  test "Test load/dump StreamLargeCSR":
    var datasetStream = newStreamCSRDataset("testsample_large", 1)
    check datasetLarge.shape == datasetStream.shape
    check datasetLarge.nnz == datasetStream.nnz
    checkDenseCSR(dataLarge, datasetStream)


  test "Test load/dump StreamLargeCSC":
    var datasetStream = newStreamCSCDataset("testsample_largeT", 1)
    check datasetLarge.shape == datasetStream.shape
    check datasetLarge.nnz == datasetStream.nnz
    checkDenseCSC(dataLarge, datasetStream)


  test "Test load/dump StreamLargeCSR2":
    var datasetStream = newStreamCSRDataset("testsample_largeTT", 1)
    check datasetLarge.shape == datasetStream.shape
    check datasetLarge.nnz == datasetStream.nnz
    checkDenseCSR(dataLarge, datasetStream)


  test "Test normalize l1 for CSR":
    var dataset: CSRDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSRDataset(data)
      dataNormalized = data
      normalizeL1(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l1)
      checkDenseCSR(dataNormalized, dataset)


  test "Test normalize l2 for CSR":
    var dataset: CSRDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSRDataset(data)
      dataNormalized = data
      normalizeL2(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l2)
      checkDenseCSR(dataNormalized, dataset)


  test "Test normalize linfty for CSR":
    var dataset: CSRDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSRDataset(data)
      dataNormalized = data
      normalizeLinfty(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=linfty)
      checkDenseCSR(dataNormalized, dataset)
  

  test "Test vstack for CSR":
    let dataset = vstack(toCSRDataset(data), toCSRDataset(data), toCSRDataset(data))
    checkDenseCSR(vstack(@[data, data, data]), dataset)


  test "Test slice for CSR":
    let dataset = toCSRDataset(data)
    for i in 0..<nSamples-1:
      for j in i+1..<nSamples:
        checkDenseCSR(data[i..<j], dataset[i..<j])
    
    checkDenseCSR(@[data[1], data[0], data[2]], dataset[[1, 0, 2]])
    checkDenseCSR(@[data[3], data[1]], dataset[[3, 1]])
    
    checkDenseCSR(data[1..^1], dataset[1..^1])
    checkDenseCSR(data[0..^2], dataset[0..^2])


  test "Test CSCDataset":
    var dataset: CSCDataset
    var y: seq[float]
    loadSVMLightFile("testsample.svm", dataset, y)
    checkDenseCSC(data, dataset)


  test "Test toCSC":
    var dataset = toCSCDataset(data)
    var y: seq[float]
    checkDenseCSC(data, dataset)
    var datasetCSR: CSRDataset
    loadSVMLightFile("testsample.svm", datasetCSR, y)
    checkDenseCSC(data, toCSCDataset(datasetCSR))


  test "Test normalize L1 for CSC":
    var dataset: CSCDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSCDataset(data)
      dataNormalized = data
      normalizeL1(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l1)
      checkDenseCSC(dataNormalized, dataset)


  test "Test normalize L2 for CSC":
    var dataset: CSCDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSCDataset(data)
      dataNormalized = data
      normalizeL2(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l2)
      checkDenseCSC(dataNormalized, dataset)


  test "Test normalize Linfty for CSC":
    var dataset: CSCDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSCDataset(data)
      dataNormalized = data
      normalizeLinfty(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=linfty)
      checkDenseCSC(dataNormalized, dataset)


  test "Test hstack for CSC":
    let dataset = hstack(toCSCDataset(data), toCSCDataset(data), toCSCDataset(data))
    checkDenseCSC(hstack(@[data, data, data]), dataset)


  test "Test slice for CSC":
    let dataset = toCSCDataset(data)
    for i in 0..<nSamples-1:
      for j in i+1..<nSamples:
        checkDenseCSC(data[i..<j], dataset[i..<j])

    checkDenseCSC(data[1..^1], dataset[1..^1])
    checkDenseCSC(data[0..^2], dataset[0..^2])

  removeFile("testsample.svm")
  removeFile("testsample_dumped.svm")
  removeFile("testsample")
  removeFile("testlabel")
  removeFile("testsampleT")
  removeFile("testsampleTT")

  removeFile("testsample_large.svm")
  removeFile("testsample_large")
  removeFile("testlabel_large")
  removeFile("testsample_largeT")
  removeFile("testsample_largeTT")