import tables, algorithm

type
  LabelEncoderObj[T] = object
    classes*: seq[T]
    table*: TableRef[T, int]
    invTable*: TableRef[int, T]

  LabelEncoder*[T] = ref LabelEncoderObj[T]


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
