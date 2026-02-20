#
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 2.01, 0.01)   # 0 to 2 with step 0.01
x = np.sin(2 * np.pi * t)

plt.plot(t, x, 'b')
plt.xlabel('t in sec')
plt.ylabel('x(t)')
plt.title('Plot of sin(2π t)')
plt.grid()
plt.show()
