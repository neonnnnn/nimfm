import unittest
import nimfm/dataset
import os, sequtils, sugar, math

proc checkDenseCSR(dense: seq[seq[float64]], sparse: CSRDataset) =
  let nSamples = sparse.nSamples
  let nFeatures = sparse.nFeatures
  check nSamples == len(dense)
  for i in 0..<nSamples:
    check nFeatures == len(dense[i])
  
  var nnzDense = 0
  var nnzSparse = 0
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      check dense[i][j] == sparse[i, j]
      if dense[i][j] != 0:
        nnzDense += 1

  for i in 0..<nSamples:
    for (j, val) in sparse.getRow(i):
      check dense[i][j] == val
      nnzSparse += 1

  check nnzDense == nnzSparse
  check nnzDense == sparse.nnz


proc checkDenseCSC(dense: seq[seq[float64]], sparse: CSCDataset) =
  let nSamples = sparse.nSamples
  let nFeatures = sparse.nFeatures
  check nSamples == len(dense)
  for i in 0..<nSamples:
    check nFeatures == len(dense[i])
  
  var nnzDense = 0
  var nnzSparse = 0
  for i in 0..<nSamples:
    for j in 0..<nFeatures:
      check dense[i][j] == sparse[i, j]
      if dense[i][j] != 0.0:
        nnzDense += 1

  for j in 0..<nFeatures:
    for (i, val) in sparse.getCol(j):
      check dense[i][j] == val
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
  const nFeatures: int = 6
  const nNz: int = 5
  var data: seq[seq[float64]]
  newSeq(data, 4)
  data[0] = @[0.0, 0.0, 1.0, 0.0, -4.2, 0.0]
  data[1] = @[-3.0, 0.0, 0, 0, 0.0, 0.0]
  data[2] = @[0.0, 0.0, 0, 0, 0, 0]
  data[3] = @[0.0, 0.0, 0, -5.0, 0.0, 103.2]
  let yTrue: array[nSamples, float] = [-1.0, 2.0, -10, 5.2]
  # create file
  var f = open("testsample.svm", fmWrite)
  for i in 0..<nSamples:
    f.write(yTrue[i])
    f.write(' ')
    for j in 0..<nFeatures:
      if data[i][j] != 0:
        f.write(j)
        f.write(':')
        f.write(data[i][j])
        if i < nFeatures-1:
          f.write(' ')
    f.write('\n')
  close(f)


  test "Test CSRDataset":
    var dataset: CSRDataset
    var y: seq[float]
    loadSVMLightFile("testsample.svm", dataset, y)
    checkDenseCSR(data, dataset)
  

  test "Test toCSR":
    var dataset = toCSR(data)
    checkDenseCSR(data, dataset)


  test "Test normalize l1 for CSR":
    var dataset: CSRDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSR(data)
      dataNormalized = data
      normalizeL1(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l1)
      checkDenseCSR(dataNormalized, dataset)


  test "Test normalize l2 for CSR":
    var dataset: CSRDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSR(data)
      dataNormalized = data
      normalizeL2(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l2)
      checkDenseCSR(dataNormalized, dataset)


  test "Test normalize linfty for CSR":
    var dataset: CSRDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSR(data)
      dataNormalized = data
      normalizeLinfty(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=linfty)
      checkDenseCSR(dataNormalized, dataset)
  

  test "Test vstack for CSR":
    let dataset = vstack(@[toCSR(data), toCSR(data), toCSR(data)])
    checkDenseCSR(vstack(@[data, data, data]), dataset)


  test "Test slice for CSR":
    let dataset = toCSR(data)
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
    var dataset = toCSC(data)
    var y: seq[float]
    checkDenseCSC(data, dataset)


  test "Test normalize L1 for CSC":
    var dataset: CSCDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSC(data)
      dataNormalized = data
      normalizeL1(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l1)
      checkDenseCSC(dataNormalized, dataset)


  test "Test normalize L2 for CSC":
    var dataset: CSCDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSC(data)
      dataNormalized = data
      normalizeL2(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=l2)
      checkDenseCSC(dataNormalized, dataset)


  test "Test normalize Linfty for CSC":
    var dataset: CSCDataset
    var dataNormalized: seq[seq[float64]]
    for axis in [0, 1]:
      dataset = toCSC(data)
      dataNormalized = data
      normalizeLinfty(dataNormalized, axis=axis)
      normalize(dataset, axis=axis, norm=linfty)
      checkDenseCSC(dataNormalized, dataset)


  test "Test hstack for CSC":
    let dataset = hstack(@[toCSC(data), toCSC(data), toCSC(data)])
    checkDenseCSC(hstack(@[data, data, data]), dataset)



  test "Test slice for CSC":
    let dataset = toCSC(data)
    for i in 0..<nSamples-1:
      for j in i+1..<nSamples:
        checkDenseCSC(data[i..<j], dataset[i..<j])

    checkDenseCSC(data[1..^1], dataset[1..^1])
    checkDenseCSC(data[0..^2], dataset[0..^2])
