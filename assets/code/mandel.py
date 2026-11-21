import numpy as np
import matplotlib.pyplot as plt
from glider_utils import *

if __name__ == '__main__':

    size = 1000
    X = np.linspace(-1.5,1.5,size)
    Y = np.linspace(-1.5,1.5,size)

    z0 = 0 + 1j*0.5

    nIter = 12

    X,Y = np.meshgrid(X,Y)

    m1 = np.ones((size,size)) + 1j*np.zeros((size,size))

    mag = np.zeros((size,size))

    m1 = m1*mag + z0

    mag = np.log(np.abs(m1))

    m1 = m1/np.abs(m1)

    Z = X + 1j*Y

    stab = np.zeros((1000,1000))
    potential = np.zeros((1000,1000))

    for i in range(nIter):
##        m1[stab == 0] = m1[stab==0]**3 + Z[stab==0]
        m1 = m1**2 + Z

        

##        stab[(np.abs(m1) > 2) * (stab == 0)] = i
        
##    np.save('mandelbrot.npy',stab)

    potential = 2**(-nIter) * np.log(np.abs(m1))
    fig,ax = plt.subplots()

    ax.imshow(potential,origin='lower',extent = (-2,1,-1.5,1.5),cmap='inferno')
    plt.show()
        
