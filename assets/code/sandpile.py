from glider_utils import *


def init(depth,ratio,load):

##    L = [np.random.rand(ratio**i,ratio**i)/ratio**(2*i) for i in range(depth)]
    L = [np.zeros((ratio**i,ratio**i))/ratio**(2*i) for i in range(depth)]

    L = np.array(L,dtype='object')

    L[0][0] += load

    return L


def update(L,depth,ratio,C,visc,friction,conserv):

    L1 = copy.deepcopy(L)

    frictionSum = 0
    steady=False

    for n in range(depth):
        for i in range(ratio**n):
            for j in range(ratio**n):
                if L[n][i,j] > C/(ratio**(2*n)):
                    L1[n][i,j] -= L[n][i,j]
##                    print('SOURCE || layer: %d row: %d col: %d' %(n,i,j))
                    for s in range(2):
                        if s==0 and n!=0:
                            if n == depth-1:
                                frictionSum += L[n][i,j]*(friction)
                                aE = L[n][i,j]*(1-friction)
                            else:
                                frictionSum += (1-visc)*L[n][i,j]*(friction)
                                aE = (1-visc)*L[n][i,j]*(1-friction)
                            casc=rng.dirichlet(np.ones(8))*aE
                            indx = 0
                            for a in range(-1,2):
                                for b in range(-1,2):
                                    if not(a==0 and b==0):
##                                        print('layer: %d row: %d col: %d' %(n,((ratio**n)+(i+a))%(ratio**n),((ratio**n)+(j+b))%(ratio**n)))
                                        L1[n][((ratio**n)+(i+a))%(ratio**n),((ratio**n)+(j+b))%(ratio**n)]+=casc[indx]
                                        indx+=1

                        if s==1 and n<(depth-1):
                            if n == 0:
                                frictionSum+=L[n][i,j]*(friction)
                                aE=L[n][i,j]*(1-friction)
                            else:
                                frictionSum+=visc*L[n][i,j]*(friction)
                                aE=visc*L[n][i,j]*(1-friction)
                            casc=rng.dirichlet(np.ones(ratio**2))*aE
                            indx = 0
                            for a in range(ratio*i,ratio*i+ratio):
                                for b in range(ratio*j,ratio*j+ratio):
##                                    print('layer: %d row: %d col: %d' %(n+1,a,b))
                                    L1[n+1][a,b]+=casc[indx]
                                    indx+=1
##                        elif s==1 and n == (depth-1):
##                            aE=visc*L[n][i,j]
##                            L1[0][0,0] += aE

    if conserv: L1[0][0,0] += frictionSum

    if all(np.array_equal(x,y) for x,y in zip(L.flat,L1.flat)):
        steady=True

    return L1,steady





if __name__ == '__main__':
    rng = np.random.default_rng(seed=int(time.time()*1000000))

    iterations = 1000
    depth = 5
    ratio = 3
    C = 1
    load= 2.5
##    visc=0.01
##    fric=0.01
    conserv=False

    size=1000

    stability = np.zeros((size,size))
    i=0
    for fric in np.logspace(-2,-1,size):
        j=0
        for visc in np.logspace(-3,0,size):
            L = init(depth,ratio,load)

            lTime = 0

            steady=False

##            fig,axs=plt.subplots(nrows=depth,ncols=1)
##            for n in range(depth):
##                axs[n].imshow(L[n],origin='lower',cmap='hot',vmax=C/(ratio**(2*n)),vmin=0)
##            plt.pause(0.1)

            for t in range(iterations):
                lStart = time.time()
                L,steady = update(L,depth,ratio,C,visc,fric,conserv)
                lEnd = time.time()

                lTime += lEnd-lStart

##                plt.clf()
##                axs = fig.subplots(nrows=depth,ncols=1)
##                for n in range(depth):
##                    axs[n].imshow(L[n],origin='lower',cmap='hot',vmax=C/(ratio**(2*n)),vmin=0)
##                plt.pause(0.1)

                if steady:
                    cTime = t
                    print('F: %.3f  v:%.3f  dissipate time: %d'%(fric,visc,t))
                    print('Average Loop Time: %.2f' %(lTime/t))
                    stability[i,j]=np.copy(t)
                    break
                elif t == iterations-1:
                    print('F: %.3f  v:%.3f  dissipate time: %d'%(fric,visc,t))
                    print('Average Loop Time: %.2f' %(lTime/t))
                    stability[i,j]=np.copy(t)
            j+=1
        i+=1

    fig,ax=plt.subplots()

    ax.imshow(stability,origin='lower',extent=[0.001,1,0.001,0.1],cmap='jet')
##    ax.set_xscale('log')
##    ax.set_yscale('log')

    ax.set_ylabel('Friction')
    ax.set_xlabel('Viscosity')
    ax.set_aspect(1)
    plt.show()
    
    print('Time to steady state: %d' %cTime)


    

        

        








    
