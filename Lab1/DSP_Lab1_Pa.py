import numpy as np
import matplotlib.pyplot as plt

def impseq(n0, n1, n2):
    n = np.arange(n1, n2+1)
    x = (n == n0).astype(int)
    return x, n

def stepseq(n0, n1, n2):
    n = np.arange(n1, n2+1)
    x = (n >= n0).astype(int)
    return x, n
## Practice a)

# n = np.arange(-5, 6)

# x1, _ = impseq(-2, -5, 5)
# x2, _ = impseq(4, -5, 5)

# x = 2*x1 - x2

# plt.figure()
# plt.stem(n, x)
# plt.title("Sequence in Problem (a)")
# plt.xlabel("n")
# plt.ylabel("x(n)")
# plt.grid()
# plt.show()

# Practice b

# n = np.arange(0, 21)

# u0, _ = stepseq(0, 0, 20)
# u10, _ = stepseq(10, 0, 20)
# u20, _ = stepseq(20, 0, 20)

# x1 = n * (u0 - u10)
# x2 = 10 * np.exp(-0.3*(n-10)) * (u10 - u20)

# x = x1 + x2

# plt.figure()
# plt.stem(n, x)
# plt.title("Sequence in Problem (b)")
# plt.xlabel("n")
# plt.ylabel("x(n)")
# plt.grid()
# plt.show()

#Practice c 

# n = np.arange(0, 51)

# w = np.random.randn(len(n))   # Gaussian noise
# x = np.cos(0.04*np.pi*n) + 0.2*w

# plt.figure()
# plt.stem(n, x)
# plt.title("Sequence in Problem (c)")
# plt.xlabel("n")
# plt.ylabel("x(n)")
# plt.grid()
# plt.show()

# Practice d

# n = np.arange(-10, 10)

# base = np.array([5,4,3,2,1])
# period = len(base)

# # Generate periodic extension
# x = base[(n % period)]

# plt.figure()
# plt.stem(n, x)
# plt.title("Sequence in Problem (d)")
# plt.xlabel("n")
# plt.ylabel("x(n)")
# plt.grid()
# plt.show()

# Example 2.2 a
n = np.arange(-2, 11)
x = np.array([1,2,3,4,5,6,7,6,5,4,3,2,1])

# helper function for shifting

def sigshift(x, n, k):
    return x, n + k

# Shift signals
x1a, n1a = sigshift(x, n, 5)     # x(n-5)
x1b, n1b = sigshift(x, n, -4)    # x(n+4)

# Find common index range
n_min = min(n1a.min(), n1b.min())
n_max = max(n1a.max(), n1b.max())
n_common = np.arange(n_min, n_max+1)

# zero-padding alignment
# helper function for shifting
def sigshift(x, n, k):
    return x, n + k

# Shift signals
x1a, n1a = sigshift(x, n, 5)     # x(n-5)
x1b, n1b = sigshift(x, n, -4)    # x(n+4)

# Find common index range
n_min = min(n1a.min(), n1b.min())
n_max = max(n1a.max(), n1b.max())
n_common = np.arange(n_min, n_max+1)

# zero-padding alignment
def align(x, nx, n_common):
    y = np.zeros(len(n_common))
    for i, val in enumerate(nx):
        idx = np.where(n_common == val)[0]
        if len(idx):
            y[idx[0]] = x[i]
    return y

x1a_aligned = align(x1a, n1a, n_common)
x1b_aligned = align(x1b, n1b, n_common)

x1 = 2*x1a_aligned - 3*x1b_aligned

plt.figure()
plt.stem(n_common, x1)
plt.title("x1(n) = 2x(n-5) - 3x(n+4)")
plt.xlabel("n")
plt.ylabel("x1(n)")
plt.grid()
plt.show()

# Example 2.2b

# folding
x_fold = x[::-1]
n_fold = -n[::-1]

# shift by 3
x_part1, n_part1 = sigshift(x_fold, n_fold, 3)


x_shift2, n_shift2 = sigshift(x, n, 2)
n_min = min(n.min(), n_shift2.min())
n_max = max(n.max(), n_shift2.max())
n_common = np.arange(n_min, n_max+1)

x_aligned = align(x, n, n_common)
x_shift2_aligned = align(x_shift2, n_shift2, n_common)

x_part2 = x_aligned * x_shift2_aligned
# align first part
x_part1_aligned = align(x_part1, n_part1, n_common)

x2 = x_part1_aligned + x_part2

plt.figure()
plt.stem(n_common, x2)
plt.title("x2(n) = x(3-n) + x(n)x(n-2)")
plt.xlabel("n")
plt.ylabel("x2(n)")
plt.grid()
plt.show()

# Example 2.3
import numpy as np
import matplotlib.pyplot as plt

# Generate n
n = np.arange(-10, 11)

# Define alpha
alpha = -0.1 + 0.3j

# Generate signal
x = np.exp(alpha * n)

# Create 4 subplots
plt.figure(figsize=(10,8))

# Real part
plt.subplot(2,2,1)
plt.stem(n, np.real(x))
plt.title("Real Part")
plt.xlabel("n")
plt.grid()

# Imaginary part
plt.subplot(2,2,2)
plt.stem(n, np.imag(x))
plt.title("Imaginary Part")
plt.xlabel("n")
plt.grid()

# Magnitude
plt.subplot(2,2,3)
plt.stem(n, np.abs(x))
plt.title("Magnitude")
plt.xlabel("n")
plt.grid()

# Phase (convert to degrees if needed)
plt.subplot(2,2,4)
plt.stem(n, np.angle(x))
