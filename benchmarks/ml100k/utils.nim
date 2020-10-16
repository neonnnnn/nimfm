import nimfm

# P: (nFeatures, nComponents)
# Too slow.
# By using a hash table, one can compute this more efficiently
proc countInteractions*(P: Matrix): int =
  result = 0
  for j1 in 0..<P.shape[0]-1:
    for j2 in j1+1..<P.shape[0]:
      if dot(P[j1], P[j2]) != 0.0:
        inc(result)


proc countFeatures*(P: Matrix): int =
  result = 0
  for j1 in 0..<P.shape[0]:
    var isSelected = 0
    for j2 in 0..<P.shape[0]:
      if j1 != j2 and dot(P[j1], P[j2]) != 0.0:
        isSelected = 1
        break
    inc(result, isSelected)