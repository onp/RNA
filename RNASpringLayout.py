import numpy as np
from scipy.spatial.distance import pdist, squareform

class SpringLayout:
    def __init__(self,reachability,posInit=None, width=600,height=600,pointsize=0.04):
        self.gCnst   = 100
        self.eCnst   = 10000
        self.sCnst   = 0.01
        self.sLength = 30
        self.damping = 0.98

        self.setbounds(width,height)
        self.rch = np.asarray(reachability[0])
        self.numPts = self.rch[0].shape[0]
        self.size = pointsize

        self.colors = np.array([[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0]])

        if len(reachability) ==3:
            self.colors = np.array([[0,0,1],[1,0,0],[0,1,0]])
            reachability.append(reachability.pop(1))

        posInit = np.asarray(posInit)
        if posInit.shape != (self.numPts,2):
            angles = np.linspace(0,2*np.pi,self.numPts,endpoint=False)
            posInit = np.vstack((np.cos(angles),np.sin(angles))).transpose()*self.bounds[0,1]/1.4

        self.clicked  = -1
        self.state    = np.hstack((posInit,np.zeros((self.numPts,2))))      # (x,y,vx,vy) for each node

        self.chgWeights(reachability)

    def chgWeights(self,newRch):
        self.rch      = np.zeros((self.numPts,self.numPts))
        self.pairs    = np.zeros((0,2))
        self.segments = np.zeros((0,2,2))
        self.segColor = np.zeros((0,8))

        for i,x in enumerate(newRch):
            x = np.asarray(x)
            x = x+x.transpose()
            self.rch += x*(20**i)
            xpairs    = np.transpose(np.nonzero(np.triu(x)))         #unique node-number pairs for existing connections
            xsegments = np.asarray([[self.state[p1,:2],self.state[p2,:2]] for p1,p2 in xpairs])
            xsegAlpha = np.transpose(np.triu(x)[np.nonzero(np.triu(x))])/x.max()        #make alpha proportional to connection strength
            xsegColor = np.hstack( (  np.ones((xsegments.shape[0],3))*self.colors[i,:].transpose()   ,   xsegAlpha[:,np.newaxis]  ) )
            xsegColor = np.hstack((xsegColor,xsegColor))

            self.pairs    = np.vstack((self.pairs,xpairs))
            self.segments = np.vstack((self.segments,xsegments))
            self.segColor = np.vstack((self.segColor,xsegColor))

        self.rch /= self.rch.max()


    def setbounds(self,width,height):
        self.bounds = np.array([-0.5,0.5])*(np.array((width,height))[:,np.newaxis])

    def step(self,dt):
        distance = squareform(pdist(self.state[:, :2]))
        np.fill_diagonal(distance,1)        #prevents NaN error
        direction = self.state[:, :2]-self.state[:,np.newaxis,:2]

        eForce = direction/(distance**3)[...,np.newaxis]
        eForce = np.sum(eForce, axis=1) * self.eCnst

        sForce = -direction*(self.rch*(distance-self.sLength))[...,np.newaxis]
        sForce = np.sum(sForce, axis=1) * self.sCnst

        gForce  =  self.state[:,:2]/np.abs(np.sum(self.state[:2]**2)) *self.gCnst

        self.state[:, 2:] -= (eForce+gForce+sForce) *dt

        self.state[:, 2:] *= self.damping

        # cap velocity
        np.putmask(self.state[:,2:],self.state[:,2:]>20,20)
        np.putmask(self.state[:,2:],self.state[:,2:]<-20,-20)

        #freeze the clicked point:
        if self.clicked != -1:
            self.state[self.clicked,:]=self.clickedState

        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]
        np.putmask(self.state[:,:2],self.state[:,:2]<self.bounds[:,0],self.bounds[:,0])
        np.putmask(self.state[:,:2],self.state[:,:2]>self.bounds[:,1],self.bounds[:,1])


        #find line positions
        self.segments =np.asarray([[self.state[p1,:2],self.state[p2,:2]] for p1,p2 in self.pairs])


    def freezeClosest(self,x,y):

        dist = np.sum((self.state[:,:2]-np.array([x,y]))**2,axis=1)
        closest = np.argmin(dist)
        if dist[closest]<100:
            self.clicked = closest
            self.clickedState = self.state[closest,:].copy()
            self.clickedState[2:] = np.zeros((1,2))

        return closest

    def moveClicked(self,dx,dy):
        if self.clicked != -1:
            self.clickedState[:2]  += np.array([dx,dy])

    def releaseClicked(self):
        self.clicked = -1




