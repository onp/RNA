import numpy as np
import scipy.linalg as linalg
import itertools
from math import sqrt


class RNAssNetwork:
    def __init__(self,sequence,weights=None,struct=None):
        sequence = sequence.upper()
        self.seqSize = len(sequence)
        self.epochCount = 0
        if weights is not None:
            self.rcInhibit,self.knotInhib,self.diagStim = weights
        else:
            self.rcInhibit = -2/self.seqSize**2   #works for size 3
            self.knotInhib = -1/self.seqSize**2
            self.diagStim  = .4
            self.learnCst  = 0.4

##            self.rcInhibit = -1/self.seqSize**1.5   #works for size 3
##            self.knotInhib = -1/self.seqSize**1.5
##            self.diagStim  = .5
##            self.learnCst  = 0.4

        baseDict = {'U':1,'G':2,'C':3,'A':4}
        matData  = itertools.product(sequence,repeat=2)
        matData = [baseDict[x[0]]+baseDict[x[1]] for x in matData]

        self.bonds = np.array(matData).reshape((self.seqSize,self.seqSize))

        #set bonds[i,j] = 1 for all watson-crick pairings i,j
        self.bonds = (self.bonds==5)*0.5    # +(conn==3)   #This would also consider G-U wobble pairs

        self.bonds = np.triu(self.bonds,4)      #removes loops smaller than 3 bp
        self.backBone = np.zeros((self.seqSize,self.seqSize))
        self.backBone[range(self.seqSize-1),range(1,self.seqSize)] = 1     #backbone


        #creating the interactions matrix (for use with row-based triu)
        diags = [(np.ones((x,x))-np.eye(x))*self.rcInhibit for x in range(self.seqSize-4,0,-1)]

        self.interactions = linalg.block_diag(*diags).astype(float)

        rOffset=0
        cOffset=0

        for d in range(1,self.seqSize-4):            #blockset diagonal - equivalent to row difference in the adjacency matrix
            diags = []                          #holder for the blocks for the diagonal

            for cn in range(self.seqSize-4-d,0,-1):                          #height for the block
                block = np.zeros((cn,cn+d))                                  #initialize the block
                block[range(cn),  range(d,cn+d)  ]  = self.rcInhibit         #inhibition along the block's diagonal d
                if d<4:
                    block[range(cn-d),range(2*d,cn+d)]  = self.diagStim/d        #stimulation along the block's diagonal d*2
                block += np.tril(np.ones((cn,cn+d))*self.knotInhib, d-1)     #knot inhibition below the block's diagonal d
                if d>4 :
                    block[:,d-4].fill(self.rcInhibit)   #complementary row/column
                    block[:,0:(d-4)].fill(0)        #These aren't knots
                diags.append(block)

            diagBlocks = linalg.block_diag(*diags).astype(float)     #create block diagonal matrix for this diagonal

            rOffset += self.seqSize-3-d
            cOffset -= d

            self.interactions[rOffset:,:cOffset] +=diagBlocks
            self.interactions[:cOffset,rOffset:] +=diagBlocks.transpose()


        self.vecInd  = np.triu_indices_from(self.bonds,4)                  #indices to pull the working vector out of the adjacency matrix
        self.bondMask = self.bonds[np.triu_indices_from(self.bonds,4)]*2         #a mask that allows for only accepted pairings
        self.bonds0 = (self.bonds*2).copy()     #the initial structure

        if struct is not None:
            self.graph = [self.backBone,self.bonds,np.array(struct)]
            print(self.graph[0].shape)
            print(self.graph[1].shape)
            print(self.graph[2].shape)
        else:
            self.graph = [self.backBone,self.bonds]


    def actFunc(self,x):
        out = np.tanh(x)       #sigmoid, output on (0,1)
        #np.putmask(out,np.random.rand(*x.shape)<(0.2/(self.epochCount+1)),1)  #randomly set some to 1
        np.putmask(out,out<0,0)
        return out


    def saveBonds(self):
        np.savez('RNAconnectivity.npz',bonds=self.bonds,chain=self.backBone)

    def epoch(self):
        activation = self.actFunc(np.dot(self.interactions,self.bonds[self.vecInd]))* self.bondMask
        self.bonds[self.vecInd] = (1-self.learnCst)*self.bonds[self.vecInd] +self.learnCst*activation
        self.epochCount += 1

    def run(self,epochs):
        for x in range(epochs):
            self.epoch()
        self.prune()
        self.saveBonds()

    def prune(self):
        np.round(self.bonds,out=self.bonds)


class RNAlearner:
    def __init__(self):
        self.rcInhibit = -1/self.seqSize**1.5   #works for size 3
        self.knotInhib = -1/self.seqSize**1.5
        self.diagStim  = .4
        self.learnCst  = 0.5
        self.weights = [self.rcInhibit,self.knotInhib,self.diagStim]

    def train(self,trainingSet,numEpochs=100):
        for seq,struct in trainingSet:
            net = RNAssNetwork(seq,weights,struct)
            net.run(numEpochs)
            weightChg = self.delta(struct,net.bonds)
            self.rcInhibit += weightChg[0]
            self.knotInhib += weightChg[1]
            self.diagStim  += weightChg[2]

    def delta(self,bPlus,bMinus,nu=0.01):
        rcDel = 0
        knotDel = 0
        diagDel = 0
        vecInd  = np.array(np.triu_indices_from(bPlus,4)).transpose()
        for indI,i in enumerate(vecInd):
            for j in vecInd[(indI+1):]:
                d = nu* (bPlus[tuple(i)]*bPlus[tuple(j)]  -  bMinus[tuple(i)]*bMinus[tuple(j)])
                if (i[0] == j[0]) or (i[1] == j[1]):
                    rcDel -= d
                elif (j[0]<i[1]) and (i[1]<j[1]):
                    knotDel -= d
                elif sum(i) == sum(j):
                    diagDel -= d
        return [rcDel,knotDel,diagDel]


if __name__  == '__main__':
    #seq = 'GGGCCCGUAGCUCAGCCAGGACAGAGCGCCGGCCUUCUAAGCCGGUGCUGCCGGGUUCAAAUCCCGGCGGGCCCGCCA'
    #seq = 'AAACCCAUGCAUAGGGUUUG'
    seq = 'AACUAAGUU'

    rnaNet1 = RNAssNetwork(seq)
    rnaNet1.run(20)
