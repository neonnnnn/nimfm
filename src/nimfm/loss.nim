import math

type
  Squared* = ref object

  SquaredHinge* = ref object

  Logistic* = ref object


  Huber* = ref object
    threshold: float64


proc newSquared*(): Squared = new(Squared)


proc loss*(self: Squared, y, p: float64): float64 = 0.5 * (y-p)^2


proc dloss*(self: Squared, y, p: float64): float64 = p-y


proc ddloss*(self: Squared, y, p: float64): float64 = 1.0


proc mu*(self: Squared): float64 = 1.0


proc newSquaredHinge*(): SquaredHinge = new(SquaredHinge)


proc loss*(self: SquaredHinge, y, p: float64): float64 = max(1-p*y, 0)^2


proc dloss*(self: SquaredHinge, y, p: float64): float64 =
  let z = 1-p*y
  if z > 0: result = -2*y*z
  else: result = 0.0


proc ddloss*(self: SquaredHinge, y, p: float64): float64 =
  let z = 1-p*y
  if z > 0: result = 2.0
  else: result = 0.0


proc mu*(self: SquaredHinge): float64 = 2.0


proc newLogistic*(): Logistic = new(Logistic)


proc loss*(self: Logistic, y, p: float64): float64 =
  let z = p * y
  if z > 0:
    result = ln(1+exp(-z))
  else:
    result = ln(exp(z)+1) - z


proc dloss*(self: Logistic, y, p: float64): float64 =
  let z = p * y
  if z > 0:
    result =  -y * exp(-z) / (1+exp(-z))
  else:
    result =  -y / (exp(z)+1)


proc ddloss*(self: Logistic, y, p: float64): float64 =
  let z = p*y
  if z > 0:
    result =  exp(-z) / ((1+exp(-z))^2)
  else:
    result = exp(z) / ((1+exp(z))^2)


proc mu*(self: Logistic): float64 = 0.25


proc newHuber*(threshold=1.0): Huber = Huber(threshold: threshold)


proc loss*(self: Huber, y, p: float64): float64 =
  let z = abs(y - p)
  if z < self.threshold: result = 0.5 * z^2
  else: result = self.threshold * (z - 0.5*self.threshold)


proc dloss*(self: Huber, y, p: float64): float64 =
  let z = abs(y-p)
  if z < self.threshold: result = y - p
  else: result = self.threshold


proc ddloss*(self: Huber, y, p: float64): float64 =
  let z = abs(y-p)
  if z < self.threshold: result = 1.0
  else: result = 0.0


proc mu*(self: Huber): float64 = 1.0
