import csv

from numpy import sin

global X
global Y

def read_data(filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            X.append(float(row['n']))
            Y.append(float(row['t']))

def f(x):
    return sin(x)

def wkx(k, x):
    p = 1
    for i in range(1, k + 1):
        p = p*(x-X[i])
    return p

def rr(k):
    S = 0
    for i in range(1, k + 1):
        p = 1
        for j in range(1, k + 1):
            if(j != i):
                p = p*(X[i]-X[j])
        S += Y[i]/p
    return S

def Nn(x, N):
    S = Y[0]
    for k in range(1, N + 1):
        S = S + wkx(k-1, x)*rr(k)
    return S

N = len(X)
x = X[0]
h = (X[N] - X[0])/(20*N)

for j in range(1, 20*N + 1):
    pass

# Використання
print("x:", X)
print("y:", Y)