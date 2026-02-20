import numpy as np
import matplotlib.pyplot as plt

plt.figure()

# ---- First Plot (Continuous)
t = np.arange(0, 2.01, 0.01)
x1 = np.sin(2 * np.pi * t)

plt.subplot(2, 1, 1)
plt.plot(t, x1)
plt.xlabel('t in sec')
plt.ylabel('x(t)')
plt.title('Plot of sin(2π t)')
plt.grid()

# ---- Second Plot (Discrete)
n = np.arange(0, 41, 1)
x2 = np.sin(0.1 * np.pi * n)

plt.subplot(2, 1, 2)
markerline, stemlines, baseline = plt.stem(n, x2)
plt.setp(markerline, markersize=4)

plt.xlabel('n')
plt.ylabel('x(n)')
plt.title('Stem Plot of sin(0.2π n)')
plt.grid()

plt.tight_layout()
plt.show()
