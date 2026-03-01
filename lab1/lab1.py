import requests
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Запит до Open-Elevation API
# -------------------------------

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]

n = len(results)

# -------------------------------
# 2. Табуляція
# -------------------------------

print("№ | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | "
          f"{point['longitude']:.6f} | "
          f"{point['elevation']:.2f}")

# -------------------------------
# 3. Кумулятивна відстань
# -------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

print("\n№ | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

def progonka(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_[i-1]
        c_[i] = c[i] / denom if i < n-1 else 0
        d_[i] = (d[i] - a[i] * d_[i-1]) / denom

    x = np.zeros(n)
    x[-1] = d_[-1]

    for i in reversed(range(n-1)):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x

def cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)

    a = y.copy()
    alpha = np.zeros(n)

    for i in range(1, n-1):
        alpha[i] = (3/h[i])*(a[i+1]-a[i]) - (3/h[i-1])*(a[i]-a[i-1])

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    B[0] = 1
    B[-1] = 1

    for i in range(1, n-1):
        A[i] = h[i-1]
        B[i] = 2*(h[i-1] + h[i])
        C[i] = h[i]
        D[i] = alpha[i]

    c = progonka(A, B, C, D)


    b = np.zeros(n-1)
    d = np.zeros(n-1)

    for i in range(n-1):
        b[i] = (a[i+1]-a[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
        d[i] = (c[i+1]-c[i])/(3*h[i])
    print("Progonka: \n a:{} \nb:{} \nc:{} \nd:{}".format(A, b, c, d))

    return a[:-1], b, c[:-1], d


def choose_points(distances, elevations, k):
    indices = np.linspace(0, len(distances)-1, k, dtype=int)
    return np.array(distances)[indices], np.array(elevations)[indices]

node_counts = [11, 16, 21]

plt.figure(figsize=(10, 6))

for k in node_counts:
    x_k, y_k = choose_points(distances, elevations, k)

    print("{}:".format(k))

    a_s, b_s, c_s, d_s = cubic_spline(x_k, y_k)

    print("\n A: {} \n B: {} \n C: {} \n D: {}".format( a_s, b_s, c_s, d_s))

    xx = np.linspace(x_k[0], x_k[-1], 500)
    yy = []

    for x_val in xx:
        for i in range(len(x_k) - 1):
            if x_k[i] <= x_val <= x_k[i + 1]:
                dx = x_val - x_k[i]
                y_val = (a_s[i] +
                         b_s[i] * dx +
                         c_s[i] * dx ** 2 +
                         d_s[i] * dx ** 3)
                yy.append(y_val)
                break

    plt.plot(xx, yy, label=f"{k-1} вузлів")

plt.plot(distances, elevations, 'ko', markersize=3, label="Усі точки")
plt.legend()
plt.xlabel("Distance")
plt.ylabel("Height")
plt.grid(True)
plt.show()

#--------------------------------

plt.figure(figsize=(10, 6))

for k in node_counts:
    x_k, y_k = choose_points(distances, elevations, k)
    a_s, b_s, c_s, d_s = cubic_spline(x_k, y_k)

    xx = np.linspace(x_k[0], x_k[-1], 500)

    y_spline_full = []

    for x_val in distances:
        for i in range(len(x_k) - 1):
            if x_k[i] <= x_val <= x_k[i + 1]:
                dx = x_val - x_k[i]
                y_val = (a_s[i] +
                         b_s[i] * dx +
                         c_s[i] * dx ** 2 +
                         d_s[i] * dx ** 3)
                y_spline_full.append(y_val)
                break

    error = np.abs(np.array(elevations) - np.array(y_spline_full))
    plt.plot(distances, error, label=f"{k-1} вузлів(похибка)")

plt.legend()
plt.xlabel("Distance")
plt.ylabel("Height Error")
plt.grid(True)
plt.show()

print("Загальна довжина маршруту (м):", distances[-1])

total_ascent = sum(max(elevations[i]-elevations[i-1],0) for i in range(1,n))
print("Сумарний набір висоти (м):", total_ascent)

total_descent = sum(max(elevations[i-1]-elevations[i],0) for i in range(1,n))
print("Сумарний спуск (м):", total_descent)

grad_full = np.gradient(yy, xx) * 100
print("Максимальний підйом (%):", np.max(grad_full))
print("Максимальний спуск (%):", np.min(grad_full))
print("Середній градієнт (%):", np.mean(np.abs(grad_full)))

mass = 80
g = 9.81
energy = mass * g * total_ascent

print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy/1000)