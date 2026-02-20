import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0, 41, 1)
x = np.sin(0.1 * np.pi * n)

markerline, stemlines, baseline = plt.stem(n, x)
plt.setp(markerline, markersize=4)

plt.xlabel('n')
plt.ylabel('x(n)')
plt.title('Stem Plot of sin(0.2π n)')
plt.grid()
plt.show()
