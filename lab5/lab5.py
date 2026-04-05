import numpy as np
import matplotlib.pyplot as plt
# Задана функція
def f(x):
 return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

def FSum(start, N, s, a, b):
    sum = 0.0
    h = (b-a)/N
    for i in range(start, N, s):
        sum += f(a+h*i)
    return sum

def Simpson(N, a, b):
    h = (b-a)/N
    return h/3*(f(a)+4*FSum(1, N, 2, a, b)+2*FSum(2, N, 2, a, b))




# Інтервал
x = np.linspace(0, 24, 1000)
y = f(x)
# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (год)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True)
plt.legend()
plt.show()