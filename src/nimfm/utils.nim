import tables, algorithm

type
  LabelEncoderObj = object
    classes*: seq[int]
    table*: TableRef[int, int]
    invTable*: TableRef[int, int]

  LabelEncoder* = ref LabelEncoderObj


proc newLabelEncoder*(): LabelEncoder =
  result = new(LabelEncoder)
  result.table = newTable[int, int]()
  result.invTable = newTable[int, int]()
  result.classes = newSeq[int]()


proc fit*(le: LabelEncoder, y: openArray[int]): LabelEncoder {.discardable.} =
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
  return le


proc transform*(le: LabelEncoder, y: openArray[int], yEnc: var seq[int]) =
  yEnc = newSeq[int](len(y))
  for i, val in y:
    if not le.table.haskey(val):
      raise newException(KeyError, "Key " & $val & " is unknown.")
    yEnc[i] = le.table[val]


proc transformed*(le: LabelEncoder, y: openArray[int]): seq[int] =
  transform(le, y, result)


proc inverseTransform*(le: LabelEncoder, y: openArray[int],
                       yEnc: var seq[int]) =
  yEnc.setLen(len(y))
  for i, val in y:
    if not le.invTable.haskey(val):
      raise newException(KeyError, "Label " & $val & " is unknown.")
    yEnc[i] = le.invTable[val]


proc inverseTransformed*(le: LabelEncoder, y: openArray[int]): seq[int] =
  inverseTransform(le, y, result)
