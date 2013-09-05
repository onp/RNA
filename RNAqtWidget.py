#!/usr/bin/env python

import sys
import math
from PySide import QtCore, QtGui, QtOpenGL

from RNASpringLayout import SpringLayout
from RNAssNetwork import RNAssNetwork
from tableviews import *

import numpy as np
import ctypes
import pyglet.gl as gl
import pyglet
import pickle as pkl

class RNAmainWindow(QtGui.QWidget):
    def __init__(self,parent,sequence,width=800,height=800,struct=None):
        QtGui.QWidget.__init__(self, parent)

        self.glWidget = GLWidget(self,sequence,width,height,struct=struct)
        self.structView = None
        self.weightView = None

        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.glWidget)
        self.setLayout(mainLayout)

        instructions = QtGui.QLabel('<b>SPACE</b> to advance an epoch; <b>P</b> to prune weak connections')
        mainLayout.addWidget(instructions)

        mvBtns = QtGui.QWidget()
        mvBtnLayout = QtGui.QHBoxLayout()
        mvBtns.setLayout(mvBtnLayout)
        mainLayout.addWidget(mvBtns)

        showWeights = QtGui.QPushButton("Show Weight Matrix")
        showWeights.clicked.connect(self.showWeights)
        mvBtnLayout.addWidget(showWeights)

        showStruct = QtGui.QPushButton("Show Structure Matrix")
        showStruct.clicked.connect(self.showStruct)
        mvBtnLayout.addWidget(showStruct)

        self.animBtn = QtGui.QCheckBox("Animate")
        self.animBtn.clicked.connect(self.animTog)
        mvBtnLayout.addWidget(self.animBtn)

        self.setWindowTitle(self.tr("SpringLayout"))

        self.timer    = QtCore.QTimer()
        self.timer.timeout.connect(self.glWidget.update)
        self.timer.setInterval(0)
        self.timer.start()

    def showWeights(self):
        if self.weightView is not None:
            self.weightView.show()
        else:
            self.weightView = WeightTableView(data = self.glWidget.secStruct.interactions)
            self.weightView.show()

    def showStruct(self):
        if self.structView is not None:
            self.structView.show()
        else:
            self.structView = StructTableView(
                    bonds = self.glWidget.secStruct.bonds,
                    interactions = self.glWidget.secStruct.interactions)
            self.structView.show()

    def animTog(self):
        self.glWidget.anim = self.animBtn.isChecked()



class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self,parent,sequence,width,height,struct=None):
        QtOpenGL.QGLWidget.__init__(self, parent)

        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.secStruct = RNAssNetwork(sequence,struct=struct)

        self.spl = SpringLayout(self.secStruct.graph,width=width,height=height)
        self.centre = np.array([width,height])/2
        self.width  = width
        self.height = height
        self.anim = False

    def minimumSizeHint(self):
        return QtCore.QSize(100, 100)

    def sizeHint(self):
        return QtCore.QSize(self.width,self.height)

    def initializeGL(self):
        gl.glClearColor(0, 0, 0, 0)
        gl.glPointSize(10.0)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

    def resizeGL(self, width, height):
        print('resize')
        self.centre = np.array([width,height])/2
        self.width,self.height = width,height
        self.spl.setbounds(width,height)

        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastX,self.lastY = event.x(),event.y()
        x,y = event.x()-self.centre[0],self.centre[1] - event.y()
        print(self.spl.freezeClosest(x,y))

    def mouseReleaseEvent(self,event):
        self.spl.releaseClicked()

    def mouseMoveEvent(self, event):
        dx = event.x()  - self.lastX
        dy = self.lastY - event.y()
        self.lastX,self.lastY = event.x(),event.y()
        self.spl.moveClicked(dx,dy)

    def keyPressEvent(self,event):
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self.secStruct.epoch()
            self.spl.chgWeights(self.secStruct.graph)
        elif key == QtCore.Qt.Key_P:
            self.secStruct.prune()
            self.spl.chgWeights(self.secStruct.graph)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()

        #draw lines
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        self.lineVerts = (self.spl.segments+self.centre).flatten()
        self.lineVerts_gl = (gl.GLfloat * len(self.lineVerts))(*self.lineVerts)
        self.lineColors = (self.spl.segColor).flatten()
        self.lineColors_gl = (gl.GLfloat * len(self.lineColors))(*self.lineColors)

        gl.glVertexPointer(2, gl.GL_FLOAT,0, self.lineVerts_gl)
        gl.glColorPointer(4,gl.GL_FLOAT,0,self.lineColors_gl)
        gl.glDrawArrays(gl.GL_LINES, 0, len(self.lineVerts) //2)

        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        #draw points
        gl.glColor3f (1.0, 1.0, 1.0)
        vertPoints = (self.spl.state[:,:2]+self.centre).flatten().astype(ctypes.c_float)
        vertices_gl = vertPoints.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, vertices_gl)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(vertPoints) // 2)

        #overdraw selected point
        if self.spl.clicked != -1:
            gl.glColor3f (1.0, 0.0, 0.0)
            point = (self.spl.clickedState[:2]+self.centre).astype(ctypes.c_float)
            point_gl = point.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, point_gl)
            gl.glDrawArrays(gl.GL_POINTS, 0, 1)

        pyglet.text.Label("Epoch: %s"%self.secStruct.epochCount,x=5,y=10,
                          font_name='Times New Roman',font_size=20).draw()

    def update(self,dt=1/30):
        if self.anim:
            self.spl.step(dt)
        self.updateGL()





if __name__ == '__main__':
    #seq = 'GGGCCCGUAGCUCAGCCAGGACAGAGCGCCGGCCUUCUAAGCCGGUGCUGCCGGGUUCAAAUCCCGGCGGGCCCGCCA'   #an actual sequence
    seq = 'UGCGGGGUGGAGCAGUcuGGCAGCUCGUCGGGCUCAUAACCCGAAGGuCGUAGGUUCAAAUCCUACCCCCGCUA'  #also real tRNA
    #seq = 'GGGCCCGUAGCUCAGCCAGGACAGAGGUUCAAAUCCCGGCGGGCCCGCCA'
    #seq = 'AAACCCAUGCAUAGGGUUUG'
    #seq = 'AACUAAGUU'

    with open('RNAstrands.rna',mode='rb') as file:
        rnaData = pkl.load(file)
    seq = rnaData[2][0]
    struct = rnaData[2][1]
    print(len(seq))
    print(len(struct))

    app = QtGui.QApplication(sys.argv)
    window = RNAmainWindow(None,seq,struct=struct)
    window.show()
    sys.exit(app.exec_())
