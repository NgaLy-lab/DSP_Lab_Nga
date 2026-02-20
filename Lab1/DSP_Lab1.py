#exapmle 1
import numpy as np

a = 3  # scalar

x = np.array([1, 2, 3])           # 1D array (row-like)
y = np.array([[1], [2], [3]])     # column vector (2D)

A = np.array([[1, 2, 3],
              [4, 5, 6]])         # matrix
print("Example 1: vector, column, matrix is \n")
print ("x row vector is",x, "\n y column vector is \n",y, "\nA matrix is \n", A )
# Example 2
print("\n==Example 2:for loop with range 1 to 6 is \n")
for i in range(1, 6):
    print(i)

# Example 3-Case 1 -Two Nested Loops (Direct Translation)
import matplotlib.pyplot as plt 
import time

start_time1 = time.time()  #   
t = np.arange(0, 1.01, 0.01)   # include 1
N = len(t)
xt1 = np.zeros(N) # generate xt with zero of length N

for n in range(N): #loop 1
    temp = 0
    for k in range(1, 4): #loop 2
        temp += (1/k) * np.sin(2 * np.pi * k * t[n])
    xt1[n] = temp

end_time1=time.time()
exectime1=end_time1-start_time1
#print(f'Run time of case 1: {exectime1} second')

plt.plot(t, xt1)
plt.title(f'DSP_Lab1-Example3_Case1: Two Nested Loops with run time: {exectime1} second')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
plt.show()



# Case 2 — One Loop (Vectorized Over Time)
#t = np.arange(0, 1.01, 0.01)
xt2 = np.zeros_like(t)

start_time2=time.time()
for k in range(1, 4):
    xt2 += (1/k) * np.sin(2 * np.pi * k * t)

end_time2=time.time()
exectime2=end_time2-start_time2

plt.plot(t, xt2)
plt.title(f'DSP_Lab1-Example3_Case2: One loop (Vectorized Over Time): {exectime2} second')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
plt.show()

# Case 3 — Fully Vectorized (Matrix Multiplication Style)
#t = np.arange(0, 1.01, 0.01)
k = np.arange(1, 4).reshape(-1, 1)   # column vector

start_time3=time.time()
# Broadcasting
xt = np.sum((1/k) * np.sin(2 * np.pi * k * t), axis=0)

end_time3=time.time()
exectime3=end_time3-start_time3
plt.plot(t, xt)
plt.title(f'DSP_Lab1-Example3_Case3: Fully Vectorized (Matrix Multiplication Style): {exectime3} second')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
plt.show()
minEx=0.0
case=3
if exectime3>exectime1:
    case=1
if exectime3 > exectime2:
    case=2
minEx=min(exectime1,exectime2,exectime3)
print(f'exec time is {minEx} with case {case}')
