import math

type
  LossFunctionKind* = enum
    Squared = "Squared",
    SquaredHinge = "SquaredHinge",
    Logistic = "Logistic"

  LossFunctionObj* = object of RootObj
    kind: LossFunctionKind
    mu*: float64

  LossFunction* = ref LossFunctionObj


proc newLossFunction*(kind: LossFunctionKind): LossFunction =
  var mu: float64
  case kind
  of Squared: mu = 1.0
  of SquaredHinge: mu = 2.0
  of Logistic: mu = 0.25
  result = LossFunction(kind: kind, mu: mu)


proc loss*(lossFunc: LossFunction, p, y: float64): float64 =
  case lossFunc.kind
  of Squared: return 0.5 * (p-y)^2
  of SquaredHinge:
    let z = 1 - p*y
    if z > 0:
      return z*z
    else:
      return 0
  of Logistic:
    let z = p*y
    if z > 0:
      return ln(1+exp(-z))
    else:
      return ln(exp(z)+1) - z


proc dloss*(lossFunc: LossFunction, p, y: float64): float64 =
  case lossFunc.kind
  of Squared: return p-y
  of SquaredHinge:
    let z = 1 - p*y
    if z > 0: return -2 * y * z
    else: return 0
  of Logistic:
    let z = p * y
    if z > 0:
      return -y * exp(-z) / (1+exp(-z))
    else:
      return y*exp(z)/(exp(z)+1) - y
