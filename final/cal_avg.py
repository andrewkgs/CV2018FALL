import sys
import numpy as np

num = np.genfromtxt(sys.argv[1])
print(num.mean())
