import numpy as np
import matplotlib.pyplot as plt
# PART B – PROBLEM 2-Random

# Define impulse & step
def delta(n):
    return (n == 0).astype(int)
def step(n):
    return (n >= 0).astype(int)

# x₁(n) Uniform [0,2]
x1 = np.random.uniform(0,2,100000)

#x₂(n) Gaussian (mean=10, var=10)
x2 = np.random.normal(10,np.sqrt(10),10000)

#Sum of Adjacent Samples: x3(n)=x1(n)+x1(n−1)
x3 = x1[:-1] + x1[1:]

# Sum of Four Independent Uniform Variables
 
y = np.random.uniform(-0.5,0.5,(4,10000))
x4 = np.sum(y,axis=0)

#Plotting
plt.figure(figsize=(10,8))

data = [x1,x2,x3,x4]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(data[i],bins=100)
    plt.title(f"Histogram {i+1}")

plt.tight_layout()
plt.show()

