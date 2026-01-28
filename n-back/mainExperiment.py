################ main.py ########################
# Written by Liam Booth 18/02/2023              #
#################################################
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
import pylsl
import random, os
import numpy as np
from threading import Thread
from playsound import playsound # python -m pip install playsound==1.2.2

class Window(QtWidgets.QMainWindow): # QWindow also required to handle keypress events properly
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.mainExperiment = MainExperiment(self) 
        self.setStyleSheet("background-color:black;")
        self.showMaximized()
        self.setCentralWidget(self.mainExperiment) 

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.deleteLater()
        elif event.key() == QtCore.Qt.Key_Space: # set keypress to true in experiment
            self.mainExperiment.keyPress = True
            print("got space")

class MainExperiment(QtWidgets.QTabWidget):    
    commandStrings = ["1-back", "2-back", "3-back", "4-back"]
    currentMarker = ""
    currentTriggerCount = 0
    triggerCounts = [100,100,100,100] #[100,100,100,100]
    ASCIILetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    testTriggerCounts = [20,20,20,20] #[20,20,20,20]
    numberBack = 1
    remember = False
    rememberCount = 0
    prevLetter = ""
    desiredLetter = ""
    tutorial = True
    keyPress = False
    stepsArray = [0,0,0,0,0]

    def __init__(self, parent):        
        super(MainExperiment, self).__init__(parent)
        self.buttonBeginRoutine = QtWidgets.QPushButton('Begin Tutorial')
        self.buttonBeginRoutine.clicked.connect(self.BeginRoutine)
        self.buttonBeginRoutine.setStyleSheet("color: white")
        self.buttonBeginRoutine.setFont(QFont('Times', 36))
        self.text = QtWidgets.QLabel("In this experiment you will be presented a series of letters multiple \n "
                                     "times. Before each series you will be instructed to press the spacebar  \n "
                                     "whenever a letter is the same as that presented so many steps back. \n \n"
                                     "First you will be shown a tutorial for all the four levels of difficulty \n"
                                     "where you are informed if you gave a correct or incorrect response. \n"
                                     "You will not be informed in the actual experiment. \n \n"
                                     "Please keep your head as still as possible during the experiment\n"
                                     "and minimise your movements to the required keyboard presses.\n\n"
                                     "You will have a 1 minute blank period to start. \n "
                                     "Please keep your attention focused on x in the middle of the monitor for this minute.\n\n"
                                     "Please inform the experimenter if you are unsure of anything or \n"
                                     "press the begin tutorial text below to begin.")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.text.setFont(QFont('Times', 38))
        self.text.setStyleSheet("color: white")
        self.setStyleSheet("background-color:black;")
        self.showMaximized()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addWidget(self.buttonBeginRoutine)
        markerInfo = pylsl.StreamInfo("n-backMarkers", 'Markers', 1, 0, 'string', 'UoH')
        self.markerOutlet = pylsl.StreamOutlet(markerInfo)
        self.correct_sound = os.path.join(os.path.dirname(__file__), "Correct.mp3")
        self.incorrect_sound = os.path.join(os.path.dirname(__file__), "Incorrect.mp3")

    def BeginRoutine(self): 
        if self.tutorial:
            self.markerOutlet.push_sample(["Started tutorial n-back"])
        else:    
            self.markerOutlet.push_sample(["Started n-back"])
        self.buttonBeginRoutine.hide()
        self.text.setFont(QFont('Times', 80))
        self.triggerCount = 0
        self.keyPress = False
        if self.tutorial:
            self.text.setText("x")
            QTimer.singleShot(60000, self.DoNewNBackRoutine)
        else:
            self.text.setText("")
            QTimer.singleShot(1000, self.DoNewNBackRoutine)
    
    def DoRoutine(self):
        self.text.setFont(QFont('Times', 110))
        x = np.random.uniform(low=0, high=1)
        if (self.remember == False):
            if (x < 0.5): # psuedo 40% of trials yield target answers
                self.remember = True
            else:
                self.remember = False
            self.rememberCount = 0    
        else:
            self.rememberCount += 1
        letter = random.choice(self.ASCIILetters)
        if (self.rememberCount == 0) and (self.remember):
            self.desiredLetter = letter
        if (self.remember == True) and (self.rememberCount >= self.numberBack):
            self.target = True
            self.remember = False
            letter = self.desiredLetter
        else:
            self.target = False
            if self.rememberCount != 0:
                while(letter == self.desiredLetter):
                    letter = random.choice(self.ASCIILetters)
                    #print("stuck1")
                    while(letter == self.prevLetter):
                        letter = random.choice(self.ASCIILetters)
                        #print("stuck2")
        print(str(self.remember) +" "+ str(self.rememberCount)+" "+ str(self.numberBack) +" "+str(self.currentTriggerCount) + " " +self.desiredLetter +" "+letter)
        self.text.setText(letter)
        self.stepsArray.append(letter)
        self.stepsArray.pop(0)
        self.prevLetter = letter
        if self.tutorial: # run routine but base of tutorial trial counts, allows custom ending
            if self.currentTriggerCount <= self.testTriggerCounts[self.numberBack-1]:
                QTimer.singleShot(1000, self.DoBaselineRoutine)
            else:
                self.currentTriggerCount = 0
                self.numberBack += 1
                if self.numberBack <= 4:
                    self.text.setText("")
                    QTimer.singleShot(1000, self.DoNewNBackRoutine)
            if self.numberBack > 4: # end experiment here
                self.tutorial = False
                self.text.setFont(QFont('Times', 50))
                self.text.setText("Tutorial Complete! \n Press the Begin Experiment text below when ready.")
                self.buttonBeginRoutine.show()
                self.buttonBeginRoutine.setText("Begin Experiment")
                self.numberBack = 1
                self.rememberCount = 0
                self.markerOutlet.push_sample(["Finished tutorial n-back"])
        else: # runs actual experiment count and ending
            if self.currentTriggerCount <= self.triggerCounts[self.numberBack-1]:
                QTimer.singleShot(1000, self.DoBaselineRoutine)
            else:
                self.currentTriggerCount = 0
                self.numberBack += 1
                if self.numberBack <= 4:
                    self.text.setText("")
                    QTimer.singleShot(1000, self.DoNewNBackRoutine)        
            if self.numberBack > 4: # end experiment here
                self.tutorial = True
                self.text.setFont(QFont('Times', 50))
                self.text.setText("Experiment finished! \n Please notify the Experimenter.")
                self.numberBack = 1
                self.markerOutlet.push_sample(["Finished n-back"])
        self.currentTriggerCount += 1

    def DoNewNBackRoutine(self):
        self.text.setFont(QFont('Times', 80))
        if self.numberBack == 1:
            self.text.setText("Remember "+str(self.numberBack)+ " step back")
        else:
            self.text.setText("Remember "+str(self.numberBack)+ " steps back")
        self.keyPress = False
        self.triggerCount = 0
        self.stepsArray = [0,0,0,0,0]
        QTimer.singleShot(6000, self.DoRoutine)

    def DoBaselineRoutine(self):
        self.text.setText("")
        if self.stepsArray[3-self.numberBack] == self.desiredLetter:
            match = True
        else:
            match = False
            #print(self.target)
        if self.tutorial:# and self.triggerCount >= self.numberBack:
            if self.keyPress == self.target:
                #Thread(target=playsound, args=(os.path.dirname(__file__)+'\eyes_open.mp3',), daemon=True).start()
                Thread(target=playsound, args=(self.correct_sound,), daemon=True).start()
            else:
                Thread(target=playsound, args=(self.incorrect_sound,), daemon=True).start()    
        self.markerOutlet.push_sample(["Steps:"+str(self.numberBack)+" KeyPress:"+str(self.keyPress)+" Matched:"+str(match)])
        print("Steps:"+str(self.numberBack)+" KeyPress:"+str(self.keyPress)+" Target:"+str(self.target))
        self.keyPress = False
        self.triggerCount += 1
        QTimer.singleShot(1000, self.DoRoutine)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
