proc comb*(n, m: int, k=0): seq[seq[int]] =
  result = @[]
  if m == 1:
    for i in k..<n:
      result.add(@[i])
  else:
    for i in k..<(n-m+1):
      for val in comb(n, m-1, i+1):
        result.add(@[i] & val)


proc combNotj*(n, m, j: int, k=0): seq[seq[int]] =
  result = @[]
  if m == 1:
    for i in k..<n:
      if i != j:
        result.add(@[i])
  else:
    for i in k..<(n-m+1):
      if i != j:
        for val in combNotj(n, m-1, j, i+1):
          result.add(@[i] & val)

