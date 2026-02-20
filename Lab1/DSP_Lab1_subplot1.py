import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(8,6))

# Continuous
t = np.arange(0, 2.01, 0.01)
ax[0].plot(t, np.sin(2*np.pi*t))
ax[0].set_title("Plot of sin(2π t)")
ax[0].set_xlabel("t in sec")
ax[0].set_ylabel("x(t)")
ax[0].grid()

# Discrete
n = np.arange(0, 41)
markerline, stemlines, baseline = ax[1].stem(n, np.sin(0.1*np.pi*n))
plt.setp(markerline, markersize=4)

ax[1].set_title("Stem Plot of sin(0.2π n)")
ax[1].set_xlabel("n")
ax[1].set_ylabel("x(n)")
ax[1].grid()

plt.tight_layout()
plt.show()
