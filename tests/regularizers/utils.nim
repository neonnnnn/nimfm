import math, nimfm/tensor/tensor, sequtils, random


proc softthreshold*(x, alpha: float64): float64 {.inline.} =
  result = float64(sgn(x)) * max(abs(x) - alpha, 0.0)


proc projL1ball*(v: var Vector, z: float64) {.inline.} =
  let n = len(v)
  var
    rho, nG, nL, offset, nCandidates, pivotIdx: int
    cumsum, cumsumCache, pivot, theta: float64
    candidates = toSeq(0..<2*n)
  nCandidates = n
  while nCandidates != 0:
    pivot_idx = candidates[offset+rand(nCandidates-1)]
    pivot = v[pivot_idx]
    nG = 0
    nL = 0
    cumsum_cache = 0
    for i in 0..<nCandidates:
      let j = candidates[offset+i]
      if j != pivotIdx:
        if v[j] >= pivot:
          cumsum_cache += v[j]
          candidates[nG] = j
          nG += 1
        else:
          candidates[n+nL] = j
          nL += 1
    # discard greaters from candidates
    if ((cumsum + cumsumCache) - float(rho+nG)*pivot) < z:
      nCandidates = nL
      offset = n
      cumsum += cumsum_cache + pivot
      candidates[nG] = pivot_idx
      nG += 1
      rho += nG
    else: # discard lessers from candidates
      nCandidates = nG
      offset = 0

  theta = (cumsum - z) / float(rho)
  for i in 0..<n:
    v[i] = v[i] - theta
    if v[i] < 0:
        v[i] = 0
