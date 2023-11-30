# This Python file uses the following encoding: utf-8
import sys
from PySide2.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QDialog, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
import sys
from PIL import Image
from PyQt5.QtGui import QPixmap

class gui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mosso Diagnostic')
        self.setWindowIcon(QtGui.QIcon("logo.jpg"))
        self.setStyleSheet("background-color: white;border: 0px solid grey")
        self.setGeometry(350, 120, 1100, 800)
        self.label = QLabel("Mosso Diagnostic")
        self.label.setFont(QFont('Bold', 40))
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.label1 = QLabel(" ")
        self.label1.setAlignment(Qt.AlignCenter | Qt.AlignCenter)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.btn1 = QPushButton("Upload")
        self.btn1.setFont(QFont('Bold', 20))
        self.btn1.setStyleSheet("background-color: white;border: 3px solid grey")
        self.btn1.clicked.connect(self.getImage)
        self.x = self.btn1
        if self.btn1.clicked == True:
            self.x=QLabel("Processing your information")
            self.x.setFont(QFont('Bold', 20))
        grid = QGridLayout()
        grid.setSpacing(20)
        grid.addWidget(self.label, 0,0,1,0)
        grid.addWidget(self.label1, 10,0,10,0)
        grid.addWidget(self.x, 0,0,4,5)
        self.setLayout(grid)
        self.show()
    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',"c://", "Image files (*.jpg *.gif)")
        imagePath = fname[0]
        basewidth = 450
        img = Image.open(imagePath)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(imagePath)
        pixmap = QPixmap(imagePath)
        self.label1.setPixmap(QPixmap(pixmap))



if __name__ == "__main__":
    app = QApplication([])
    window = gui()
    window.show()
    sys.exit(app.exec_())

