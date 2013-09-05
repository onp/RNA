# a first pyside table

import sys
from PySide import QtCore, QtGui

import numpy as np


class WeightTableModel(QtCore.QAbstractTableModel):
    def __init__(self,a):
        QtCore.QAbstractTableModel.__init__(self)
        self.dataArray=a
        self.nrmP = np.max(self.dataArray)
        self.nrmN = 1.4*np.min(self.dataArray)

    def rowCount(self,parent=None):
        return self.dataArray.shape[0]

    def columnCount(self,parent=None):
        return self.dataArray.shape[1]

    def data(self,index,role):
        val = self.dataArray[index.row(),index.column()]
        if role == QtCore.Qt.DisplayRole:
            return '%.2f'%val
        elif role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignCenter
        elif role == QtCore.Qt.BackgroundRole:
            if val<0:
                c = 255*(1-val/self.nrmN)
                return QtGui.QColor(255,c,c)
            elif val>0:
                c = 255*(1-val/self.nrmP)
                return QtGui.QColor(c,255,c)


class WeightTableView(QtGui.QWidget):
    def __init__(self,parent=None,data=None):
        QtGui.QWidget.__init__(self,parent)
        self.table = QtGui.QTableView(parent)
        if data is not None:
            self.model_ = WeightTableModel(data)
            self.table.setModel(self.model_)
        self.changeSize(35)

        v_box = QtGui.QVBoxLayout()
        self.setLayout(v_box)
        self.sizeCtrl = QtGui.QSpinBox()
        self.sizeCtrl.setMaximum(50)
        self.sizeCtrl.setValue(35)
        self.sizeCtrl.valueChanged.connect(self.changeSize)
        v_box.addWidget(self.sizeCtrl)
        v_box.addWidget(self.table)


    def changeSize(self,newSize):
        for x in range(self.model_.columnCount()):
            self.table.setColumnWidth(x,newSize)
        for x in range(self.model_.rowCount()):
            self.table.setRowHeight(x,newSize)
        self.model_.layoutChanged.emit()


class StructTableModel(QtCore.QAbstractTableModel):
    def __init__(self,view,bonds,interactions):
        QtCore.QAbstractTableModel.__init__(self)
        self.view = view
        self.bondData = bonds
        self.interactionData = interactions
        self.vecInd  = np.triu_indices_from(bonds,4)
        self.colors = np.zeros_like(bonds)
        self.dispMode = 'Struct'
        self.nrmB = np.max(self.bondData)
        self.nrmP = np.max(self.interactionData)
        self.nrmN = 1.4*np.min(self.interactionData)

    def rowCount(self,parent=None):
        return self.bondData.shape[0]

    def columnCount(self,parent=None):
        return self.bondData.shape[1]

    def selectionChg(self,index=None):
        r = index.row()
        c = index.column()
        if r>c:
            r,c = c,r
        m = self.bondData.shape[1]
        x = sum(range(m-4,m-r-4,-1))+(c-r-4)
        self.colors[self.vecInd] = self.interactionData[x,:]
        print(self.interactionData[x,:])
        self.nrmP = np.max(self.interactionData)
        self.nrmN = 1.4*np.min(self.interactionData)
        self.nrmB = np.max(self.bondData)
        self.layoutChanged.emit()

    def data(self,index,role):
        if index.row() < index.column():
            val = self.bondData[index.row(),index.column()]
            color = self.colors[index.row(),index.column()]
        else:
            val = self.bondData[index.column(),index.row()]
            color = self.colors[index.column(),index.row()]
            if role == QtCore.Qt.DisplayRole:
                return ''

        if self.dispMode == 'Struct':
            if role == QtCore.Qt.DisplayRole:
                return '%.2g'%val
            elif role == QtCore.Qt.TextAlignmentRole:
                return QtCore.Qt.AlignCenter
            elif role == QtCore.Qt.BackgroundRole:
                c = 255*(1-val/self.nrmB)
                return QtGui.QColor(c,255,c)

        elif self.dispMode == "Interact":
            if role == QtCore.Qt.DisplayRole:
                return '%.2g'%color
            elif role == QtCore.Qt.TextAlignmentRole:
                return QtCore.Qt.AlignCenter
            elif role == QtCore.Qt.BackgroundRole:
                if val>0:
                    if color<0:
                        c = 255*(1-color/self.nrmN)
                        return QtGui.QColor(255,c,c)
                    elif color>0:
                        c = 255*(1-color/self.nrmP)
                        return QtGui.QColor(c,255,c)


class StructTableView(QtGui.QWidget):
    def __init__(self,parent=None,bonds=None,interactions=None):
        QtGui.QWidget.__init__(self,parent)
        v_box = QtGui.QVBoxLayout()
        self.setLayout(v_box)
        opt_wid = QtGui.QWidget(self)
        h_box = QtGui.QHBoxLayout()
        opt_wid.setLayout(h_box)
        v_box.addWidget(opt_wid)

        self.table = QtGui.QTableView(parent)
        self.table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        if bonds is not None:
            self.model_ = StructTableModel(self,bonds,interactions)
            self.table.setModel(self.model_)
            self.changeSize(35)

        self.structBtn = QtGui.QRadioButton("Show Structure",parent=opt_wid)
        h_box.addWidget(self.structBtn)
        self.structBtn.toggled.connect(self.onToggle)
        self.structBtn.click()
        interBtn = QtGui.QRadioButton("Show Interactions", parent=opt_wid)
        h_box.addWidget(interBtn)
        self.sizeCtrl = QtGui.QSpinBox()
        self.sizeCtrl.setMaximum(50)
        self.sizeCtrl.setValue(35)
        self.sizeCtrl.valueChanged.connect(self.changeSize)
        h_box.addWidget(self.sizeCtrl)



        self.table.clicked.connect(self.onClick)

        v_box.addWidget(self.table)

    def onClick(self,index):
        self.model_.selectionChg(index)

    def onToggle(self,checked):
        if self.structBtn.isChecked():
            self.model_.dispMode = 'Struct'
        else:
            self.model_.dispMode = 'Interact'
        self.model_.layoutChanged.emit()

    def changeSize(self,newSize):
        for x in range(self.model_.columnCount()):
            self.table.setColumnWidth(x,newSize)
        for x in range(self.model_.rowCount()):
            self.table.setRowHeight(x,newSize)
        self.model_.layoutChanged.emit()



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    anArray= np.array([[2,3,4],[5,-7,3],[2,1,-3],[8,4,5]])
    tv = WeightTableView(data=anArray)
    tv.show()
    app.exec_()
