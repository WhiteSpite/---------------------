import numpy as np  
from timer import timer

a = {1: {2: {0: 1, 1: 2}}}

b = a[1][2]
b[0]+=10

print(a)
