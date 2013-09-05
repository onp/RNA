import numpy as np
import scipy.linalg as linalg
import itertools

baseDict = {'U':1,'G':2,'C':3,'A':4}

def actFunc(x):
    out  = (  1/(1+np.exp(-x))  )-0.5
    return out
#seq = 'GGGCCCGUAGCUCAGCCAGGACAGAGCGCCGGCCUUCUAAGCCGGUGCUGCCGGGUUCAAAUCCCGGCGGGCCCGCCA'
seq = 'AAACCCAUGCAUAGGGUUUG'
#seq = 'AACUAAGUU'
seqSize = len(seq)

rcInhibit = -0.7/np.sqrt(seqSize)
knotInhib = -0.7/np.sqrt(seqSize)
diagStim  = .4
carry = -0.8*rcInhibit

matData  = itertools.product(seq,repeat=2)

matData = [baseDict[x[0]]+baseDict[x[1]] for x in matData]

conn = np.array(matData).reshape((seqSize,seqSize))

#conn =1 for all watson-crick pairings
conn = (conn==5).astype(float)*0.5    #*2 +(conn==3).astype(float)

#add backbone/exclude small loops from initialization conditions

conn = np.triu(conn,4)      #removes small loops
backBone = np.zeros((seqSize,seqSize))
backBone[range(seqSize-1),range(1,seqSize)] = 1     #backbone

#creating the interactions matrix (for use with row-based triu)
diags = [(np.ones((x,x))-np.eye(x))*rcInhibit for x in range(seqSize-4,0,-1)]

interactions = linalg.block_diag(*diags).astype(float)
np.fill_diagonal(interactions,carry)    #feedback to self

rOffset=0
cOffset=0

for d in range(1,seqSize-4):            #blockset diagonal - equivalent to row in the adjacency matrix
    diags = []                          #holder for the blocks for the diagonal

    for cn in range(seqSize-4-d,0,-1):            #height for the block
        block = np.zeros((cn,cn+d))               #initialize the block
        block[range(cn),  range(d,cn+d)  ]  = rcInhibit         #inhibition along the block's diagonal d
        block[range(cn-d),range(2*d,cn+d)]  = diagStim/d          #stimulation along the block's diagonal d*2  --this could be turned into a neighbourhood function
        block += np.tril(np.ones((cn,cn+d))*knotInhib, d-1)     #knot inhibition below the block's diagonal d
        diags.append(block)

    diagBlocks = linalg.block_diag(*diags).astype(float)     #create block diagonal matrix for this diagonal

    rOffset += seqSize-3-d
    cOffset -= d
    interactions[rOffset:,:cOffset] +=diagBlocks
    interactions[:cOffset,rOffset:] +=diagBlocks.transpose()



np.savez('RNAconnectivity.npz',conn,backBone)

vecInd  = np.triu_indices_from(conn,4)                  #indices to pull the working vector out of the adjacency matrix
connMask = conn[np.triu_indices_from(conn,4)]*2    #the mask that allows for only accepted pairings
connO = (conn*2).copy()     #the initial structure


#find structure
for x in range(100):
    activation = actFunc(np.dot(interactions,conn[vecInd]))* connMask
    allowedChg = 0.5-np.abs(conn[vecInd]-0.5)
    conn[vecInd] = conn[vecInd] +allowedChg*activation


np.savez('RNAconnectivityF.npz',conn,backBone)

