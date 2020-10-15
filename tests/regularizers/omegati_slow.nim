import nimfm/tensor/tensor, utils, ../kernels_slow


type
  OmegaTISlow* = ref object
    value*: float64


proc newOmegaTISlow*(): OmegaTISlow =
  new(result)

# for coordinate descent
proc prox*(self: OmegaTISlow, P: var Matrix, lam: float64, 
           degree, s, j: int) {.inline.} =
  let psj = P[s, j]
  var absp = zeros([len(P[s])])
  for j1 in 0..<len(absp):
    absp[j1] = abs(P[s, j1])
  absp[j] = 0.0
  var strength = lam*anovaSlow(absp, ones([len(absp)]), degree-1)
  P[s, j] = softthreshold(psj, strength)