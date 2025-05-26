def lorenz(t, y, sigma=10, rho=28, beta=8/3):
    x, y_, z = y
    dxdt = sigma * (y_ - x)
    dydt = x * (rho - z) - y_
    dzdt = x * y_ - beta * z
    return [dxdt, dydt, dzdt]

def rossler(t, y, a=0.2, b=0.2, c=5.7):
    x, y_, z = y
    dxdt = -y_ - z
    dydt = x + a * y_
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]
