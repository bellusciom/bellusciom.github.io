from glider_utils import *


if __name__ == '__main__':
    mode='icond'

    if mode in ['param']:
        w=1
        cycles = 10
        dt = np.array([0.01,0.001,0.0005,0.0001])
        tmax=cycles*2*np.pi*w
        iterations=(tmax/dt).astype('int')

        size=1000

        F = np.linspace(0.1,2,size)
        d = np.linspace(0,1,size)

        F,d = np.meshgrid(F,d)

        stability = np.zeros((size,size))

        for i in range(size):
            print('d: %.2f' %d[i,0])
            for j in range(size):
                dtIndx=0
                stable=0
                while stable == 0 and dtIndx<len(dt):
                    xTS = np.zeros((iterations[dtIndx]))
                    x = 0
                    xdot=0
                    xddot=0
                    for iteration in range(iterations[dtIndx]):
                        t= iteration * dt[dtIndx]
                        print('Time: %.3f' %t)
                        xddot1 = -d[i,j]*xdot + x - x**3 + F[i,j]*np.cos(t)
                        xdot1 = xdot + xddot*dt[dtIndx]
                        x1 = x + xdot*dt[dtIndx]

                        xddot = xddot1
                        xdot = xdot1
                        x=x1

                        if np.abs(x) > 5:
    ##                        print('Unstable Simulation F: %.2f  d: %.2f  dt: %.3f ' %(F[i,j],d[i,j],dt[dtIndx]))
                            dtIndx += 1
                            break

                        xTS[iteration] = x

                    if xTS[-1] != 0:
                        stable = 1

    ##            print('F: %.2f  d: %.2f  maxX: %.1f' %(F[i,j],d[i,j],np.max(np.abs(xTS))))

                stability[i,j] = np.max(xTS[int(iterations[dtIndx]-(2//dt[dtIndx])):])-np.min(xTS[int(iterations[dtIndx]-(2//dt[dtIndx])):])


        fig,ax = plt.subplots()
        ax.imshow(stability,origin='lower',extent=(0,2,0,1),cmap='jet')
        ax.set_xlabel('F')
        ax.set_ylabel('D')
        plt.show()
    elif mode in ['icond']:
        w=1
        cycles = 10
        dt = np.array([0.01,0.001,0.0005,0.0001])
        dtIndx=1
        tmax=cycles*2*np.pi*w
        iterations=(tmax/dt[dtIndx]).astype('int')

        F=1.25
        d=0.1

        size=1000

        X = np.linspace(-2.5,2.5,size)
        Y = np.linspace(-2.5,2.5,size)

        X,Y = np.meshgrid(X,Y)

        Z = np.zeros((size,size))

        fig,ax = plt.subplots()

        ax.imshow(X,origin='lower',extent=(-2.5,2.5,-2.5,2.5),cmap='jet')

        ax.set_ylabel('y0')
        ax.set_xlabel('x0')
        plt.pause(0.1)

        

        for iteration in range(iterations):
            t=iteration*dt[dtIndx]
            Z1 = -d*Y + X - X**3 + F*np.cos(t)
            Y1 = Y + Z*dt[dtIndx]
            X1 = X + Y*dt[dtIndx]

            Z = Z1
            Y = Y1
            X=X1

            fig.clf()

            ax = fig.subplots()

            ax.imshow(X,origin='lower',extent=(-2.5,2.5,-2.5,2.5),cmap='jet')

            ax.set_ylabel('y0')
            ax.set_xlabel('x0')
            ax.set_title('Time: %.3f' %t)
            ax.set_aspect(1)
            plt.pause(0.1)
                

            

            

                
            

    
