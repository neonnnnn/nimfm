import unittest

import nimfm/utils
suite "Test label encoder":
  let y1 = [4, -3, 100, -3, 4, 100, 100, 4]
  let y1Enc = [1, 0, 2, 0, 1, 2, 2, 1]
  let y1Classes = [-3, 4, 100]
  let y2 = [-1, -1, -1, -1, -1, -1]
  let y2Enc = [0, 0, 0, 0, 0, 0]
  let y2Classes = [-1]
  let y3 = [0, 1, 2, 3, 4, 5, 6, 7]
  let y3Enc = [0, 1, 2, 3, 4, 5, 6, 7]
  let y3Classes = [0, 1, 2, 3, 4, 5, 6, 7]

  test "Test transform":
    var le = newLabelEncoder()
    var yEnc: seq[int]
    le.fit(y1)
    check le.transformed(y1) == y1Enc
    le.transform(y1, yEnc)
    check yEnc == y1Enc

    le.fit(y2)
    check le.transformed(y2) == y2Enc
    le.transform(y2, yEnc)
    check yEnc == y2Enc

    le.fit(y3)
    check le.transformed(y3) == y3Enc
    le.transform(y3, yEnc)
    check yEnc == y3Enc

  test "Test inverse transform":
    var le = newLabelEncoder()
    var yEnc: seq[int]

    le.fit(y1)
    check le.inverseTransformed(y1Enc) == y1
    le.inverseTransform(y1Enc, yEnc)
    check yEnc == y1

    le.fit(y2)
    check le.inverseTransformed(y2Enc) == y2
    le.inverseTransform(y2Enc, yEnc)
    check yEnc == y2

    le.fit(y3)
    check le.inverseTransformed(y3Enc) == y3
    le.inverseTransform(y3Enc, yEnc)
    check yEnc == y3

  test "Test classes":
    var le = newLabelEncoder()

    le.fit(y1)
    check le.classes == y1Classes
    le.fit(y2)
    check le.classes == y2Classes
    le.fit(y3)
    check le.classes == y3Classes
