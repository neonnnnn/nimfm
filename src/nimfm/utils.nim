import tables, algorithm, sequtils, math

type
  LabelEncoderObj[T] = object
    classes*: seq[T]
    table*: TableRef[T, int]
    invTable*: TableRef[int, T]

  LabelEncoder*[T] = ref LabelEncoderObj[T]


proc argsort*[T](a: T, order=SortOrder.Ascending): seq[int] =
  result = toSeq(0..<a.len)
  case order
  of SortOrder.Ascending: 
    sort(result,  proc(i, j: int ): int = cmp(a[i], a[j]))
  of SortOrder.Descending:
    sort(result,  proc(i, j: int ): int = -cmp(a[i], a[j]))


proc argmin*[T](a: seq[T]): int =
  result = 0
  for i in 1..<len(a):
    if a[i] < a[result]: result = i


proc argmax*[T](a: seq[T]): int =
  result = 0
  for i in 1..<len(a):
    if a[i] > a[result]: result = i


proc expit*(x: float64): float64  =  exp(min(0.0, x)) / (1.0 + exp(-abs(x)))


proc expit*(a: openarray[float64]): seq[float64] = a.map(expit)    


proc newLabelEncoder*[T](): LabelEncoder[T] =
  result = new(LabelEncoder[T])
  result.table = newTable[T, int]()
  result.invTable = newTable[int, T]()
  result.classes = newSeq[T]()


proc fit*[T](le: LabelEncoder[T], y: openArray[T]) =
  # initialization
  var nClasses: int = 0
  clear(le.table)
  clear(le.invTable)
  le.classes.setLen(0)

  for val in y:
    if not le.table.hasKey(val):
      le.table[val] = nClasses
      le.classes.add(val)
      inc(nClasses)
  le.classes.sort()
  # sorted transformation
  for i, val in le.classes:
    le.table[val] = i
    le.invTable[i] = val


proc transform*[T](le: LabelEncoder[T], y: openArray[T], yEnc: var seq[int]) =
  yEnc = newSeq[int](len(y))
  for i, val in y:
    if not le.table.haskey(val):
      raise newException(KeyError, "Key " & $val & " is unknown.")
    yEnc[i] = le.table[val]


proc transformed*[T](le: LabelEncoder[T], y: openArray[T]): seq[int] =
  transform(le, y, result)


proc inverseTransform*[T](le: LabelEncoder[T], y: openArray[int],
                          yEnc: var seq[T]) =
  yEnc.setLen(len(y))
  for i, val in y:
    if not le.invTable.haskey(val):
      raise newException(KeyError, "Label " & $val & " is unknown.")
    yEnc[i] = le.invTable[val]


proc inverseTransformed*[T](le: LabelEncoder[T], y: openArray[int]): seq[T] =
  inverseTransform(le, y, result)
