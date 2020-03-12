import unittest, sugar, sequtils, math
import nimfm/metrics

suite "Test metrics":
  let yTrue = @[-1, -1, 1, 1]
  let yTrue01 = @[0, 0, 1, 1]
  let yScore1 = @[0.1, 0.4, 0.35, 0.8] # tp=1, fp=0, tn=2, fn=1
  let yScore2 = @[-0.1, 0.1, 0.9, -0.2] # tp=1, fp=1, tn=1, fn=1
  let prec1 =1.0
  let recall1 = 0.5
  let fscore1 = 1.0/1.5
  let prec2 = 0.5
  let recall2 = 0.5
  let fscore2 = 0.5
  let zeros = @[0.0, 0.0, 0.0, 0.0]
  let ones = @[1.0, 1.0, 1.0, 1.0]
  let inverse = @[1, 1, -1, -1]
  let inverse01 = @[1, 1, 0, 0]


  test "Test roc auc":
    check rocauc(yTrue, yScore1) == 0.75
    check rocauc(yTrue, yScore2) == 0.5
    check rocauc(yTrue, yTrue.map(x=>toFloat(x))) == 1
    check rocauc(yTrue, zeros) == 0.5
    check rocauc(yTrue, ones) == 0.5
    check rocauc(yTrue, inverse.map(x=>toFloat(x))) == 0
    check rocauc(yTrue, inverse01.map(x=>toFloat(x))) == 0

    check rocauc(yTrue01, yScore1) == 0.75
    check rocauc(yTrue01, yScore2) == 0.5
    check rocauc(yTrue01, yTrue.map(x=>toFloat(x))) == 1
    check rocauc(yTrue01, zeros) == 0.5
    check rocauc(yTrue01, ones) == 0.5
    check rocauc(yTrue01, inverse.map(x=>toFloat(x))) == 0
    check rocauc(yTrue01, inverse01.map(x=>toFloat(x))) == 0


  test "Test accuracy":
    check accuracy(yTrue, yScore1.map(x=>sgn(x-0.5))) == 0.75
    check accuracy(yTrue, yScore2.map(x=>sgn(x))) == 0.5
    check accuracy(yTrue, yTrue) == 1.0
    check accuracy(yTrue, zeros.map(x=>toInt(x))) == 0
    check accuracy(yTrue, ones.map(x=>toInt(x))) == 0.5

    check accuracy(yTrue01, yScore1.map(x=>int((sgn(x-0.5)+1)/2))) == 0.75
    check accuracy(yTrue01, yScore2.map(x=>int((sgn(x)+1)/2))) == 0.5
    check accuracy(yTrue01, yTrue01) == 1
    check accuracy(yTrue, zeros.map(x=>toInt(x))) == 0
    check accuracy(yTrue, ones.map(x=>toInt(x))) == 0.5


  test "Test precision, recall, f-score":
    var actual: (float, float, float)
    actual = precisionRecallFscore(yTrue, yScore1.map(x=>sgn(x-0.5))) 
    check actual == (prec1, recall1, fscore1)
    actual = precisionRecallFscore(yTrue, yScore2.map(x=>sgn(x))) 
    check actual ==  (prec2, recall2, fscore2)
    actual = precisionRecallFscore(yTrue, yTrue)
    check actual == (1.0, 1.0, 1.0)
    actual = precisionRecallFscore(yTrue, zeros.map(x=>toInt(x))) 
    check actual == (0.0, 0.0, 0.0)
    actual = precisionRecallFscore(yTrue, ones.map(x=>toInt(x)))
    check actual == (0.5, 1.0, 1.0/1.5)
    actual =  precisionRecallFscore(yTrue01, yScore1.map(x=>sgn(x-0.5)))
    check actual == (prec1, recall1, fscore1)
    actual = precisionRecallFscore(yTrue01, yScore2.map(x=>sgn(x))) 
    check actual ==  (prec2, recall2, fscore2)
    check precisionRecallFscore(yTrue01, yTrue) == (1.0, 1.0, 1.0)
    actual = precisionRecallFscore(yTrue01, zeros.map(x=>toInt(x))) 
    check actual == (0.0, 0.0, 0.0)
    actual = precisionRecallFscore(yTrue01, ones.map(x=>toInt(x))) 
    check actual == (0.5, 1.0, 1.0/1.5)
    actual = precisionRecallFscore(
      zeros.map(x=>toInt(x)), zeros.map(x=>toInt(x))) 
    check actual == (0.0, 0.0, 0.0)
    actual = precisionRecallFscore(
      ones.map(x=>toInt(x)), ones.map(x=>toInt(x)))  
    check actual == (1.0, 1.0, 1.0)
    actual = precisionRecallFscore(
      zeros.map(x=>toInt(x)), zeros.map(x=>toInt(x)),
      pos=0) 
    check actual == (1.0, 1.0, 1.0)
    actual = precisionRecallFscore(
      ones.map(x=>toInt(x)), ones.map(x=>toInt(x)),
      pos=0)  
    check actual == (0.0, 0.0, 0.0)

  test "Test RMSE":
    check rmse(yScore1, yScore2) == 0.5984354601792912
    check rmse(yScore2, yScore1) == 0.5984354601792912
    check rmse(yScore1, yScore1) == 0.0
    check rmse(yScore2, yScore2) == 0.0
    check rmse(zeros, ones) == 1.0
    check rmse(ones, zeros) == 1.0
    check rmse(ones, ones) == 0.0
    check rmse(zeros, zeros) == 0.0


  test "Test r2 score":
    check r2(yScore1, yScore2) == -4.687344913151364
    check r2(yScore2, yScore1) ==  -0.9163879598662203
    check r2(yScore1, yScore1) == 1.0
    check r2(yScore2, yScore2) == 1.0
    check r2(zeros, yScore1) == 0.0
    check r2(zeros, yScore2) == 0.0
    check r2(zeros, zeros) == 1.0
    check r2(ones, yScore1) == 0.0
    check r2(ones, yScore2) == 0.0
    check r2(ones, ones) == 1.0