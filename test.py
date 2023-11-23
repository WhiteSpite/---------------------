import numpy as np  
from timer import timer
a = {}
for i in range(100000000):
    a[i] = i


with timer():
    for i in range(100000000):
        a[0]
