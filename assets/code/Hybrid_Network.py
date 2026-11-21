# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

EEStr = 20
EIStr = 20
IEStr = 10.5
IIStr = 10
LRStr = 5
iteration = 5
t=500

a = np.zeros((iteration + 2)).astype('int64')
b = np.zeros((iteration+2)).astype('int64')

for i in range(iteration+1):
    a[i+1] = 2*a[i] + 1
    b[i+1] = a[i+1] + 1
    
n = a[-1]
    
# EGrid = np.zeros((n,n))

W = np.zeros((2*(n**2),2*(n**2)))

for i in range(n):
    for j in range(n):
        for k in range(-2,3):
            for l in range(-2,3):
                iindx = (i + k + n)%n
                jindx = (j+l + n)%n
                if k == 0 and l == 0:
                    W[i*n + j,n**2 + iindx*n + jindx] = EIStr
                if np.abs(k) < 2 and np.abs(l) < 2:
                    W[i*n + j,iindx*n + jindx] = EEStr
                    W[i*n + j,n**2 + iindx*n + jindx] = EIStr
                    W[n**2 + i*n + j,n**2 + iindx*n + jindx] = -IIStr
                    W[n**2 + i*n + j,iindx*n + jindx] = -IEStr
                else:
                    W[n**2 + i*n + j,iindx*n + jindx] = -IEStr/2
                
for i in range(1,iteration-2):
    for jrow in range(2**i):
        for jcol in range(2**i):
            binjRow = bin(jrow)
            binjCol = bin(jcol)
            binRow = np.zeros(i,dtype='int64')
            binCol = np.zeros(i,dtype='int64')
            for k in range(i):
                if binjRow[-(k+1)] in ['b']: break
                binRow[k] = int(binjRow[-(k+1)])
            for k in range(i):
                if binjCol[-(k+1)] in ['b']: break
                binCol[k] = int(binjCol[-(k+1)])
                
            row1 = np.sum(binRow * np.flip(b[:-1])[:i]) + a[iteration-i]
            col1 = np.sum(binCol * np.flip(b[:-1])[:i]) + a[iteration-i]
            row0 = np.sum(binRow[:-1] * np.flip(b[:-1])[:i-1]) + a[iteration-i+1]
            col0 = np.sum(binCol[:-1] * np.flip(b[:-1])[:i-1]) + a[iteration-i+1]
            

            # W[row1*n + col1,row0*n + col0] += LRStr
            W[row0*n + col0,row1*n + col1] += LRStr
        

EGrid = np.zeros((n,n))
IGrid = np.zeros((n,n))
eRefrac = np.zeros((n,n))
iRefrac = np.zeros((n,n))

# EGrid[a[-2],a[-2]] = 1
EGrid[np.random.randint(7*n//16,9*n//16),np.random.randint(7*n//16,9*n//16)]=1

fig = plt.figure()
axs = fig.subplots(2)
axs[0].imshow(EGrid,cmap='hot',origin='lower',vmax=1)
axs[1].imshow(IGrid,cmap='hot',origin='lower',vmax=1)
plt.pause(.01)

for i in range(t):
    eRefrac[EGrid>0] = 2
    iRefrac[IGrid>0] = 2
    cGrid = np.concat([EGrid.reshape(1,n**2),IGrid.reshape(1,n**2)],axis=1)
    plt.clf()
    nGrid = np.matmul(cGrid,W)
    nGrid[nGrid<0] = 0
    # nGrid[nGrid>1] = 1
    nGrid = np.tanh(nGrid)
    EGrid = nGrid[:,:n**2].reshape(n,n)
    IGrid = nGrid[:,n**2:].reshape(n,n)
    EGrid[eRefrac>0] = 0
    IGrid[iRefrac>0] = 0
    eRefrac = eRefrac - 1
    iRefrac = iRefrac - 1
    if np.sum(EGrid) == 0:
        EGrid[np.random.randint(7*n//16,9*n//16),np.random.randint(7*n//16,9*n//16)]=1
    axs = fig.subplots(2)
    axs[0].imshow(EGrid,cmap='hot',origin='lower',vmax=1)
    axs[1].imshow(IGrid,cmap='hot',origin='lower',vmax=1)
    plt.pause(.01)


    