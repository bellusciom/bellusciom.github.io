#!/usr/bin/env python

import numpy as np
import argparse
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import colors
import os, sys, time

def fig_close(event):
     event.canvas.figure.is_closed = True


def main(size,mode,density,ptsize):

    font = ('Times',-ptsize,'bold')

    grid = mGenerate(size,density,mode)
    
    fig = plt.figure()
    fig.is_closed = False
    fig.canvas.mpl_connect('close_event', fig_close)

    draw_grid(fig,grid,size,ptsize,font,mode)
    
    start_time = time.time()
    
    while np.sum(grid[0] == 0) > 0 and np.max(grid[0] * grid[2]) < 1:
        grid = solveAlgorithm(grid,size,mode)
        draw_grid(fig,grid,size,ptsize,font,mode)
        
    if np.sum(grid[0] == 0) == 0:
        end_time = time.time()
        time_elapsed = end_time-start_time
        print('You Win Gunker')
        print('Time: %.2f' %time_elapsed)
    elif np.max(grid[0] * grid[2]) == 1:
        print("Game Over")



def mGenerate(size,density,mode):
    if mode in ['run','test']:
        bGrid = np.random.rand(size,size)<(density)
    elif mode in ['te']:
        bGrid = np.array([[0,0,0,0],[1,0,0,0],[0,0,0,0],[1,1,0,0]])

    nGrid = np.zeros((size,size))

    nGrid = nGrid + np.concatenate((np.concatenate((bGrid[1:size,1:size],np.zeros((1,size-1))),axis=0),np.zeros((size,1))), axis=1)
    nGrid = nGrid + np.concatenate((np.zeros((size,1)),np.concatenate((bGrid[1:size,0:size-1],np.zeros((1,size-1))),axis=0)), axis=1)
    nGrid = nGrid + np.concatenate((np.concatenate((np.zeros((1,size-1)),bGrid[0:size-1,1:size]),axis=0),np.zeros((size,1))), axis=1)
    nGrid = nGrid + np.concatenate((np.zeros((size,1)),np.concatenate((np.zeros((1,size-1)),bGrid[0:size-1,0:size-1]),axis=0)), axis=1)

    nGrid = nGrid + np.concatenate((bGrid[:,1:size],np.zeros((size,1))),axis=1)
    nGrid = nGrid + np.concatenate((np.zeros((size,1)),bGrid[:,0:size-1]),axis=1)
    nGrid = nGrid + np.concatenate((bGrid[1:size,:],np.zeros((1,size))),axis=0)
    nGrid = nGrid + np.concatenate((np.zeros((1,size)),bGrid[0:size-1,:]),axis=0)

    nGrid = nGrid + bGrid
    
    vGrid = np.zeros((size,size))

    grid = np.array([vGrid, nGrid, bGrid])

    return grid

def solveAlgorithm(grid,size,mode):
    bMatrix = np.zeros((0,size**2))
    N = np.zeros((0,1))
    eGrid = grid[0] * grid[1]
    moves = []
    if np.sum(grid[0]) == 0:
        if mode in ['run','test']:
            moves = [[np.random.randint(size),np.random.randint(size),1]]
        elif mode in ['te']:
            moves = [[0,3,1]]
    else:
        hidNbr = np.zeros((size,size))
        revNbr = np.zeros((size,size))
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                hidNbr[int(i == 1):size-int(i==-1),int(j==1):size-int(j==-1)] += grid[0,int(i==-1):size-int(i==1),int(j==-1):size-int(j==1)] == 0
                revNbr[int(i == 1):size-int(i==-1),int(j==1):size-int(j==-1)] += grid[0,int(i==-1):size-int(i==1),int(j==-1):size-int(j==1)] == 1
                
        boundary = ((hidNbr * revNbr) > 0) * grid[0] == 1
        
        row_ind,col_ind = np.where(boundary==1)
        
        vals = list(np.array([row_ind,col_ind]).T)
        
        for i,j in vals:
            x = np.zeros((1,size**2))
            minL = max(0,i-1)
            maxL = min(size,i+2)
            minK = max(0,j-1)
            maxK = min(size,j+2)
            numHidden = np.sum(grid[0,minL:maxL,minK:maxK] == 0)
            numFlagged = np.sum(grid[0,minL:maxL,minK:maxK] == -1)
            numBomb = grid[1,i,j]
    
            if numHidden > 0:
                if numBomb == (numHidden + numFlagged):
                    for l in range(minL,maxL):
                        for k in range(minK,maxK):
                            if grid[0,l,k] == 0 and not([l,k,-1] in moves):
                                moves = moves + [[l,k,-1]]
                elif grid[1,i,j] == numFlagged:
                    for l in range(minL,maxL):
                        for k in range(minK,maxK):
                            if grid[0,l,k] == 0 and not([l,k,1] in moves):
                                moves = moves + [[l,k,1]]
                else:
                    for L1 in range(minL,maxL):
                        for K1 in range(minK,maxK):
                            if grid[0,L1,K1] == 1 and not(L1==i and K1==j):
                                if L1 == i:
                                    sub1x1 = minL
                                    sub1x2 = maxL - 1
                                else:
                                    sub1x1 = min(L1,i)
                                    sub1x2 = max(L1,i)
                                if K1 == j:
                                    sub1y1 = minK
                                    sub1y2 = maxK - 1
                                else:
                                    sub1y1 = min(K1,j)
                                    sub1y2 = max(K1,j)
    
                                numHiddenSub1 = np.sum(grid[0,sub1x1:sub1x2+1,sub1y1:sub1y2+1] == 0)
                                if numHiddenSub1 == 0: continue
                                
                                numFlaggedSub1 = np.sum(grid[0,sub1x1:sub1x2+1,sub1y1:sub1y2+1] == -1)
    
                                numHiddenGrid1 = np.sum(grid[0,L1-1:L1+2,K1-1:K1+2] == 0)
                                numFlaggedGrid1 = np.sum(grid[0,L1-1:L1+2,K1-1:K1+2] == -1)
    
                                numHiddenSubC = numHidden - numHiddenSub1
                                numFlaggedSubC = numFlagged - numFlaggedSub1
    
                                minNumBombSubC = grid[1,i,j] - min(numHiddenSub1+numFlaggedSub1,grid[1,L1,K1])
    
                                
    
                                if numHiddenSubC + numFlaggedSubC == minNumBombSubC:
                                    if mode in ['test']:
                                        print('Grid: \n', grid[:,minL:maxL,minK:maxK])
                                        print('Num Hidden in Grid: ', numHidden)
                                        print('Num Flagged in Grid: ', numFlagged)
                                        print('Num Bombs in Grid: ', grid[1,i,j])
                                        print('Subgrid: \n', grid[:,sub1x1:sub1x2+1,sub1y1:sub1y2+1])
                                        print('Num Hidden in Subgrid: ', numHiddenSub1)
                                        print('Num Flagged in Subgrid: ', numFlaggedSub1)
                                        print('Num Bombs in Subgrid: ', grid[1,L1,K1])
                                        print('Flagging all not in Subgrid')
                                    for L in range(minL,maxL):
                                        for K in range(minK,maxK):
                                            if mode in ['test']:
                                                print('Current grid Values, Row: ', L, ' Col: ', K)
                                                print('Subgrid Range, x1: ', sub1x1, ' x2: ', sub1x2, ' y1: ', sub1y1,' y2: ',sub1y2)
                                            if grid[0,L,K] == 0 and (L < sub1x1 or L > sub1x2 or K < sub1y1 or K > sub1y2):
                                                if not([L,K,-1] in moves):
                                                    moves = moves + [[L,K,-1]]
                                                    if mode in ['test']:
                                                        print('Move Added: ', [L,K,-1])
                                elif numHiddenSub1 == numHiddenGrid1:
                                    numBombSub1 = grid[1,L1,K1] - (numFlaggedGrid1 - numFlaggedSub1)
                                    if numBombSub1 + numFlaggedSubC == grid[1,i,j]:
                                        if mode in ['test']:
                                            print('Grid: \n', grid[:,minL:maxL,minK:maxK])
                                            print('Num Hidden in Grid: ', numHidden)
                                            print('Num Flagged in Grid: ', numFlagged)
                                            print('Num Bombs in Grid: ', grid[1,i,j])
                                            print('Subgrid: \n', grid[:,sub1x1:sub1x2+1,sub1y1:sub1y2+1])
                                            print('Num Hidden in Subgrid: ', numHiddenSub1)
                                            print('Num Flagged in Subgrid: ', numFlaggedSub1)
                                            print('Num Bombs in Subgrid: ', grid[1,L1,K1])
                                            print('Revealing all not in Subgrid')
                                        for L in range(minL,maxL):
                                            for K in range(minK,maxK):
                                                if mode in ['test']:
                                                    print('Current grid Values, Row: ', L, ' Col: ', K)
                                                    print('Subgrid Range, x1: ', sub1x1, ' x2: ', sub1x2, ' y1: ', sub1y1,' y2: ',sub1y2)
                                                if grid[0,L,K] == 0 and (L < sub1x1 or L > sub1x2 or K < sub1y1 or K > sub1y2):
                                                    if not([L,K,1] in moves):
                                                        moves = moves + [[L,K,1]]
                                                        if mode in ['test']:
                                                            print('Move Added: ', [L,K,1])
                                # else:
                                    # for L2 in range(minL,maxL):
                                    #     for K2 in range(minK,maxK):
                                    #         if not(grid[0,L2,K2] == 1) or ((L1 == L2) and (K1 == K2)) or ((L2 == i) and (K2 == j)): continue
                                    #         if ((L1 == L2) and (L1 == i)) or ((K1 == K2) and (K1 == j)): continue
    
                                    #         if L2 == i:
                                    #             sub2x1 = minL
                                    #             sub2x2 = maxL - 1
                                    #         else:
                                    #             sub2x1 = min(L2,i)
                                    #             sub2x2 = max(L2,i)
                                    #         if K2 == j:
                                    #             sub2y1 = minK
                                    #             sub2y2 = maxK - 1
                                    #         else:
                                    #             sub2y1 = min(K2,j)
                                    #             sub2y2 = max(K2,j)
    
                                    #         numHiddenSub2 = np.sum(grid[0,sub2x1:sub2x2+1,sub2y1:sub2y2+1] == 0)
                                    #         if numHiddenSub2 == 0: continue
                                            
                                    #         numFlaggedSub2 = np.sum(grid[0,sub2x1:sub2x2+1,sub2y1:sub2y2+1] == -1)
    
                                    #         numHiddenGrid2 = np.sum(grid[0,L2-1:L2+2,K2-1:K2+2] == 0)
                                    #         numFlaggedGrid2 = np.sum(grid[0,L2-1:L2+2,K2-1:K2+2] == -1)
    
                                    #         numHiddenSubC = numHidden - numHiddenSub1
                                    #         numFlaggedSubC = numFlagged - numFlaggedSub1
    
                                    #         minNumBombSubC = grid[1,i,j] - min(numHiddenSub1+numFlaggedSub1,grid[1,L1,K1]) - min(numHiddenSub2+numFlaggedSub2,grid[1,L2,K2])
                                            
    
                                            
                                            
                                                                 
    while len(moves) == 0:
        i = np.random.randint(size)
        j = np.random.randint(size)
        if grid[0,i,j] == 0:
            moves = [[i,j,1]]

    if mode in ['test']:
        print('Moves: \n', moves)

    for m in moves:
        grid[0,m[0],m[1]] = m[2]
        # if m[2] == 1 and grid[2,m[0],m[1]] == 1:
        #     print('Game Over')
            
    return grid

##def solveAIInit():
##    fLayer = np.random.rand(((2/3)*(size**2),1))
##    sLayer = np.random.rand(3,1)
##
##    hLayer = [fLayer,sLayer]
##
##    return hLayer
##
##def solveAIRun(grid, hLayer):
##    while gO == 1:
##        m = []
##        for i in range(size):
##            for j in range(size):
##                if grid[i,j,0] == 0:
##                    m += [0]
##                elif grid[i,j,1] >= 0:
##                    m += [grid[i,j,1]]
##                elif grid[i,j,0] == -1:
##                    m += [-1]
##
##        m = np.array(m)
##        m = layerComp(m,hLayer)
##
##        [grid,gO] = mUpdate(m,grid)
##
##    score = np.sum(np.sum(grid[:,:,0] != 0))
##
##    return
##    
##    
##def layerComp(m,hLayer):
##    s1 = hLayer[0].shape[0]
##    s2 = hLayer[1].shape[0]
##
##    sm = m.shape[0]
##
##    L1 = []
##    for i in range(sm-s1):
##        L1 += m[0+i:s1+i] * hLayer[0]
##    m = sum(L1)
##
##    L2 = []
##    for i in range(s1-s2):
##        L2 += m[0+i:s2+i] * hLayer[1]
##
##    m = np.round(sum(L2))
##
##    return m
    


def draw_grid(fig,grid,size,ptsize,mFont,mode):
    gridDraw = np.rot90(grid,axes=(2,1))
    
    gridDraw = gridDraw[0]*(gridDraw[1] + 10*gridDraw[2] + 1)
    
    cdict = {'red': [(0,0,0),
                     (.0909,1,1),
                     (.1818,.3,.3),
                     (.9091,0,0),
                     (1,1,1)],
             'green': [(0,1,1),
                       (.0909,1,1),
                       (.1818,0,0),
                       (.9091,.7,.7),
                       (1,0,0)],
             'blue': [(0,0,0),
                      (.0909,1,1),
                      (.1818,.6,.6),
                      (.9091,1,1),
                      (1,0,0)]}

    cmap = colors.LinearSegmentedColormap('ghot',cdict)
    
    plt.clf()
    ax = fig.subplots()
    ax.imshow(gridDraw,cmap=cmap,vmax=10,vmin=-1,origin='lower')
    plt.pause(.01)

    return


if __name__ == "__main__":
    
    nargs=len(sys.argv)
      
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=30)
    parser.add_argument('--mode', type=str, default="run")
    parser.add_argument('--density', type=float, default=.15)
    parser.add_argument('--ptsize', type=int, default=20)

    args = parser.parse_args()

    main(args.size,args.mode,args.density,args.ptsize)
