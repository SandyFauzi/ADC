# numerical_methods.py
# listing kode untuk menyimpan fungsi Metode Numerik.

import numpy as np
from scipy.interpolate import interp1d

# =============================================================================
# MODUL 5: INTERPOLASI
# =============================================================================

def linear_interpolation(x_data, y_data):
    return interp1d(x_data, y_data, kind='linear', fill_value="extrapolate")

def cubic_interpolation(x_data, y_data):
    return interp1d(x_data, y_data, kind='cubic', fill_value="extrapolate")

# =============================================================================
# MODUL 6: REGRESI
# =============================================================================

def polynomial_regression(x_data, y_data, degree):
    X_list = []
    for xi in x_data:
        row = [xi**p for p in range(degree + 1)]
        X_list.append(row)
    X = np.array(X_list)

    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)
    
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    Xty = np.dot(Xt, y_data)
    
    beta = np.dot(XtX_inv, Xty)
    
    poly_func = np.poly1d(np.flip(beta))
    
    return beta, poly_func

# =============================================================================
# MODUL 7: INTEGRASI NUMERIK
# =============================================================================

def trapezoid_rule(f, a, b, n):
    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n):
        integral += 2 * f(a + i * h)
    integral *= h / 2
    return integral

def simpson_13_rule(f, a, b, n):
    if n % 2 != 0:
        n -= 1
    if n < 2:
        return 0

    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n):
        if i % 2 == 0:
            integral += 2 * f(a + i * h)
        else:
            integral += 4 * f(a + i * h)
    integral *= h / 3
    return integral

def simpson_38_rule(f, a, b, n):
    if n % 3 != 0:
        n = int(round(n / 3.0) * 3) 
    if n < 3:
        return 0

    h = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n):
        if i % 3 == 0:
            integral += 2 * f(a + i * h)
        else:
            integral += 3 * f(a + i * h)
    integral *= 3 * h / 8
    return integral

# =============================================================================
# MODUL 8: DIFERENSIASI NUMERIK
# =============================================================================

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# =============================================================================
# MODUL 9: PERSAMAAN DIFERENSIAL BIASA (PDB)
# =============================================================================

def forward_euler(f, t_span, y0, h):
    t0, tf = t_span
    n = int(abs((tf - t0) / h))
    t_values = np.linspace(t0, tf, n + 1)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0

    for i in range(n):
        y_values[i+1] = y_values[i] + h * f(t_values[i], y_values[i])
        
    return t_values, y_values

def runge_kutta_4(f, t_span, y0, h):
    t0, tf = t_span
    n = int(abs((tf - t0) / h))
    t_values = np.linspace(t0, tf, n + 1)

    y0_arr = np.array(y0, dtype=float)
    y_values = np.zeros((n + 1, y0_arr.size))
    y_values[0, :] = y0_arr

    for i in range(n):
        t_i = t_values[i]
        y_i = y_values[i, :]

        k1 = h * np.array(f(t_i, y_i))
        k2 = h * np.array(f(t_i + 0.5 * h, y_i + 0.5 * k1))
        k3 = h * np.array(f(t_i + 0.5 * h, y_i + 0.5 * k2))
        k4 = h * np.array(f(t_i + h, y_i + k3))

        y_values[i+1, :] = y_i + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return t_values, y_values.squeeze()
