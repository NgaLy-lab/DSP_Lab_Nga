import numpy as np
import matplotlib.pyplot as plt
# PART A – PROBLEM 1

# Define impulse & step

def delta(n):
    return (n == 0).astype(int)
def step(n):
    return (n >= 0).astype(int)

# Signal 1: x_1 (n)=3δ(n+2)+2δ(n)-δ(n-3)+5δ(n-7)
n1 = np.arange(-5,16)
x1 = 3*delta(n1+2) +   2*delta(n1) -   delta(n1-3) +   5*delta(n1-7)

# Signal 2: x_2 (n)=∑_(k=-5)^5▒e^(-∣k∣)  δ(n-2k)
n2 = np.arange(-10,11)
x2 = np.zeros_like(n2,dtype=float)
for k in range(-5,6):
    	   x2 += np.exp(-abs(k)) * delta(n2-2*k)

# Singal 3: x_3 (n)=10u(n)-5u(n-5)-10u(n-10)+5u(n-15)
x3 = 10*step(n1) - 5*step(n1-5)   -10*step(n1-10) + 5*step(n1-15)

# Signal 4: x_4 (n)=e^0.1n [u(n+20)-u(n-10)]
x4 = np.exp(0.1*n1)*(step(n1+20)-step(n1-10))

#Signal 5: x_5 (n)=5[cos⁡(0.49πn)+cos⁡(0.51πn)]
n5 = np.arange(-200,201)
x5 = 5*(np.cos(0.49*np.pi*n5)+   np.cos(0.51*np.pi*n5))

#Signal 6: x_6 (n)=2sin⁡(0.01πn)cos⁡(0.5πn)
x6 = 2*np.sin(0.01*np.pi*n5)* np.cos(0.5*np.pi*n5)

#Singal 7: x_7 (n)=e^(-0.05n) sin⁡(0.1πn+π/3)
n7 = np.arange(0,101)
x7 = np.exp(-0.05*n7)*  np.sin(0.1*np.pi*n7 + np.pi/3)

#Signal 8: x_8 (n)=e^0.01n sin⁡(0.1πn)
x8 = np.exp(0.01*n7)*    np.sin(0.1*np.pi*n7)

#Plotting
plt.figure(figsize=(12,10))
signals = [x1,x2,x3,x4,x5,x6,x7,x8]
indices = [n1,n2,n1,n1,n5,n5,n7,n7]

for i in range(8):
    plt.subplot(4,2,i+1)
    plt.stem(indices[i],signals[i])
    plt.title(f"x{i+1}(n)")
    plt.grid()
plt.tight_layout()
plt.show()







