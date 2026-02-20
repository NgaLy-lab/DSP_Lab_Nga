import numpy as np
import matplotlib.pyplot as plt

def impseq(n0, n1, n2):
    """
    Generates x(n) = δ(n-n0), for n1 <= n <= n2
    """
    n = np.arange(n1, n2 + 1)
    x = (n == n0).astype(int)
    return x, n


# Example
x, n = impseq(n0=3, n1=0, n2=10)

plt.stem(n, x)
plt.title("Unit Sample Sequence δ(n-3)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
