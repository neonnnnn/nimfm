import ../tensor/tensor


type
  Params* = ref object
    P*: Tensor
    w*: Vector
    intercept*: float64
    fitLinear*: bool
    fitIntercept*: bool


proc newParams*(shape_P: array[3, int], len_w: int,
                fitLinear, fitIntercept: bool): Params =
  new(result)
  result.fitLinear = fitLinear
  result.fitIntercept = fitIntercept
  result.P = zeros(shape_P)
  result.w = zeros([len_w])
  result.intercept = 0.0


proc newParams*(P: var Tensor, w: var Vector, intercept: float64,
                fitLinear, fitIntercept: bool): Params =
  new(result)
  result.fitLinear = fitLinear
  result.fitIntercept = fitIntercept
  result.P = P
  result.w = w
  result.intercept =intercept


proc add*(self: Params, grad: Params, eta_intercept, eta_w, eta_P: float64) =
  if self.P.shape != grad.P.shape:
    raise newException(ValueError, "self.P.shape != grad.P.shape.")
  for i in 0..<self.P.shape[0]:
    for j in 0..<self.P.shape[1]:
      for k in 0..<self.P.shape[2]:
        self.P[i, j, k] += eta_P * grad.P[i, j, k]
  
  if self.fitLinear and grad.fitLinear:
    if self.w.shape != grad.w.shape:
      raise newException(ValueError, "self.w.shape != grad.w.shape.")
    for i in 0..<self.w.shape[0]:
      self.w[i] += eta_w * grad.w[i]

  if self.fitIntercept and grad.fitLinear:
    self.intercept += eta_intercept * grad.intercept


proc add*(self: Params, grad: Params, eta: float64) =
  self.add(grad, eta, eta, eta)


proc `+=`*(self: Params, grad: Params) = self.add(grad, 1.0)


proc `-=`*(self: Params, grad: Params) = self.add(grad, -1.0)


proc scale*(self: Params, scale_intercept, scale_w, scale_P: float64) =
  self.P *= scale_P
  if self.fitLinear:
    self.w *= scale_w
  if self.fitIntercept:
    self.intercept *= scale_intercept


proc scale*(self: Params, scale: float64) =
  self.scale(scale, scale, scale)


proc `*=`*(self: Params, scale: float64) = self.scale(scale)


proc `/=`*(self: Params, scale: float64) = self.scale(1.0 / scale)


proc `<-`*(self: Params, params: Params) =
  self.P <- params.P
  self.w <- params.w
  self.intercept = params.intercept

proc `<-`*(self: Params, c: float64) =
  self.P <- c
  self.w <- c
  self.intercept = c


proc step*(self, grads: Params, eta_intercept, eta_w, eta_P: float64,
           alpha0, alpha, beta: float64) {.inline.} =
  let 
    scale_P = 1.0 + eta_P * beta
    scale_w = 1.0 + eta_w * alpha
    scale_intercept = 1.0 + eta_intercept * alpha0
    
  self.add(grads, -eta_intercept, -eta_w, -eta_P)
  self.scale(1.0 / scale_intercept, 1.0 / scale_w, 1.0 / scale_P)


proc dot*(params1, params2: Params): float64 =
  result = dot(params1.P, params2.P)
  if params1.fitLinear and params2.fitLinear:
    result += dot(params1.w, params2.w)
  if params1.fitIntercept and params2.fitIntercept:
    result += params1.intercept * params2.intercept
