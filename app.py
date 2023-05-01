import sys
import tensorflow as tf
from PyQt5 import QtCore, QtWidgets, QtMultimedia
from main import *
from predict import *

class WorkerSignals(QtCore.QObject):
    result = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
class Worker(QtCore.QRunnable):

    def __init__(self, fileName):
        super(Worker, self).__init__()
        self.file = fileName
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):

        print("Thread start")
        print(self.file)
        res = predict(self.file)
        self.signals.result.emit(res)  # Return the result of the processing
        self.signals.finished.emit()  # Done


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        # Select
        self.selectFileButton = QtWidgets.QPushButton("Upload .wav file")
        self.selectFileButton.clicked.connect(self.selectFile)
        # Selected info
        self.selectedFileText = QtWidgets.QLabel(text="Selected file: -")
        self.emotionText = QtWidgets.QLabel(text="Emotion: -")

        # Progress bar
        self.progressBar = QtWidgets.QProgressBar(self, minimum=0, maximum=0)
        self.progressBar.hide()

        # Media player
        self.mediaPlayer = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.LowLatency)

        self.mediaPlayer.stateChanged.connect(self.mediastateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.setSlidePosition)
        # Play button
        self.playButton = QtWidgets.QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        # Detect button
        self.classifyButton = QtWidgets.QPushButton("Classify emotion")
        self.classifyButton.clicked.connect(self.detect)

        # Vertiacal Layout and adding widgets to it
        vLayout = QtWidgets.QVBoxLayout()

        vLayout.addWidget(self.selectFileButton)
        vLayout.addWidget(self.selectedFileText)
        vLayout.addWidget(self.emotionText)
        vLayout.addWidget(self.progressBar)


        vLayout.addWidget(self.slider)
        vLayout.addWidget(self.playButton)

        vLayout.addWidget(self.classifyButton)

        # Main Window Geometry
        self.setGeometry(300, 300, 1020, 800)
        self.setFixedSize(800, 300)
        self.setLayout(vLayout)
        self.setWindowTitle("Speech Emotion Recognition")
        self.show()
        self.fileName = ""

        self.threadpool = QtCore.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def selectFile(self):
        self.selectedFileText.setText("Selected file: -")
        self.emotionText.setText("Emotion: -")
        file, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                        'Open File',
                                                        './',
                                                        'Audio Files (*.wav)')
        if not file:
            return
        else:
            self.mediaPlayer.setMedia(
                QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(file)))

        self.selectedFileText.setText("Selected file: " + file)
        self.fileName = file

    def mediastateChanged(self, state):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.slider.setValue(position)

    def durationChanged(self, duration):
        self.slider.setRange(0, duration)

    def setSlidePosition(self, position):
        self.mediaPlayer.setPosition(position)

    def play(self):
        if self.mediaPlayer.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def drawResult(self, text):
        self.emotionText.setText("Emotion: " + text)

    def threadFinished(self):
        self.selectFileButton.setEnabled(True)
        self.classifyButton.setEnabled(True)
        self.progressBar.hide()
        print("Thread finished!")

    def detect(self):
        self.selectFileButton.setEnabled(False)
        self.classifyButton.setEnabled(False)
        self.progressBar.show()
        print(self.fileName)
        worker = Worker(self.fileName)
        worker.signals.result.connect(self.drawResult)
        worker.signals.finished.connect(self.threadFinished)
        self.threadpool.start(worker)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())