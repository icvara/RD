import numpy as np
import matplotlib.pyplot as plt

A=np.arange(0,20)

R = (0 + 10*np.power(A*.1,4))/(1+np.power(A*.1,4))
R1 = (5 + 10*np.power(A*.1,4))/(1+np.power(A*.1,4))
R2 = 5 + ( 5*np.power(A*.1,4))/(1+np.power(A*.1,4))


plt.plot(R)
plt.plot(R1)
plt.plot(R2,'--')

plt.show()