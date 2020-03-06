import unittest
import nimfm/dataset
import os


suite "Test datasets":
  const nSamples: int = 4
  const nFeatures: int = 5
  const nNz: int = 5
  var data: seq[seq[float64]]
  newSeq(data, 4)
  data[0] = @[0.0, 1.0, 0.0, -4.2, 0.0]
  data[1] = @[-3.0, 0, 0, 0.0, 0.0]
  data[2] = @[0.0, 0, 0, 0, 0]
  data[3] = @[0.0, 0, -5.0, 0.0, 103.2]
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
    for i in 0..<nSamples:
      for (j, val) in dataset.getRow(i):
        check data[i][j] == val
    check dataset.nNz == nNz
    check dataset.nSamples == nSamples
    check dataset.nFeatures == nFeatures

  test "Test toCSR":
    var dataset = toCSR(data)
    var y: seq[float]
    loadSVMLightFile("testsample.svm", dataset, y)
    for i in 0..<nSamples:
      for (j, val) in dataset.getRow(i):
        check data[i][j] == val
    check dataset.nNz == nNz
    check dataset.nSamples == nSamples
    check dataset.nFeatures == nFeatures


  test "Test CSCDataset":
    var dataset: CSCDataset
    var y: seq[float]
    loadSVMLightFile("testsample.svm", dataset, y)
    for j in 0..<nFeatures:
      for (i, val) in dataset.getCol(j):
        check data[i][j] == val
    check dataset.nNz == nNz
    check dataset.nSamples == nSamples
    check dataset.nFeatures == nFeatures


  test "Test toCSC":
    var dataset = toCSC(data)
    var y: seq[float]
    loadSVMLightFile("testsample.svm", dataset, y)
    for j in 0..<nFeatures:
      for (i, val) in dataset.getCol(j):
        check data[i][j] == val
    check dataset.nNz == nNz
    check dataset.nSamples == nSamples
    check dataset.nFeatures == nFeatures


  removeFile("testsample.svm")
