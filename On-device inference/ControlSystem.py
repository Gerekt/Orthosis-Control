#commands = ['up', 'stop', 'no', 'right', 'left', 'down', 'go', 'off', 'on', 'yes', '_unknown_'] #Has to be done this way in order to keep the order of commands
commands = ['blue', 'red', '_unknown_']

import os, sys
import numpy as np
import sounddevice as sd
import time
import timeit
import queue
import threading
from scipy import signal
import tensorflow as tf
from numpy.lib.shape_base import expand_dims

from wavToSpectrogram import get_spectrogram, get_mfcc, get_tflite_spectrogram, get_scipy_spectrogram, get_spectrogram_inference

callbackTimeTracker = []
resampleTimeTracker = []
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Ignores message that a certain instruction set is not used.

def listDevices():
    print("list devices")
    print(sd.query_devices(kind='input'))

def displayTimes(arrayQueueTime, arrayWindowTime, arrayFeatureTime, arrayPredictionTime, timeStart):
    global callbackTimeTracker
    global resampleTimeTracker
    print()
    print()
    print("Total time needed for inputstream callbacks: ", np.sum(callbackTimeTracker) * 1000)
    print("Time needed for resampling in inputstream: ", np.average(resampleTimeTracker) * 1000)
    #print("Amount of callbacks: ", len(callbackTimeTracker))
    callbackTimeTracker = []
    resampleTimeTracker = []

    print("Total time needed for prediciton  is: ", (timeit.default_timer() - timeStart)* 1000)
    print()
    print("Time needed for copying audio frames from queue in ms: ", np.average(arrayQueueTime) * 1000)
    print("Time needed for windowing: ", np.average(arrayWindowTime) * 1000)
    print("Time needed for feature extraction ", np.average(arrayFeatureTime) * 1000)
    print("Time needed for prediciton: ", np.average(arrayPredictionTime) * 1000)


def loadModel():
    if os.path.isfile("model.tflite") != True:
        print("Model path not found")
        return False
    else:
        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate

        model = Interpreter(model_path="model.tflite")
        model.allocate_tensors()
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        return model, input_details, output_details

def performInference(model, featuresAudio, input_details, output_details):
    featuresAudio = tf.expand_dims(featuresAudio, axis=0)
    model.set_tensor(input_details[0]['index'], featuresAudio)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    #print(output_data)
    return commands[output_data.argmax()]

def classificationQueue(*, q, dataWindow, HL):
    print("start classificaiton queue")
    emptyCounter = 0
    arrayQueueTime = []
    arrayWindowTime = []
    arrayFeatureTime = []
    arrayPredictionTime = []

    model, input_details, output_details = loadModel()
    if model == False:
        return
    
    timeStart = time.time()
    t = threading.current_thread()

    while getattr(t, "do_run", True):
        try:
            startTime = timeit.default_timer()
            #print("Time at start prediction is: ", startTime)
            #print("get attribute succes now replacing window....")
            #print("queue length: ", q.qsize())
            #print("Queue time before get is: ", timeit.default_timer())
            data = q.get(True, 1) # Wait until full hoplength of samples is ready in queue, waits no longer than 1 sec
            #print("Queue time after get is:: ", timeit.default_timer())
            arrayQueueTime.append(timeit.default_timer() - startTime)

            #shift = len(data)
            #print("length of data is: ", shift)
            windowStartTime = timeit.default_timer()
            shift = len(data)
            dataWindow = np.roll(dataWindow, -shift)            # dataWindow is total window length over which the features are calculated
            dataWindow[-shift:] = data
            arrayWindowTime.append(timeit.default_timer() - windowStartTime)                                  # new incoming raw microphone data is added to window
            #print("Time needed to replace window in ms: ", (timeit.default_timer() - windowsStartTime)*1000)
            #print("Window replaced")
            #arrayCopyTime.append(timeit.default_timer() - windowsStartTime)
            #displayTimes(arrayCopyTime, HL, timeStart)
            #arrayCopyTime = [] 
            
            featureStartTime = timeit.default_timer()
            featuresAudio = get_scipy_spectrogram(dataWindow)
            #featuresAudio = get_mfcc(dataWindow)
            arrayFeatureTime.append(timeit.default_timer() - featureStartTime)
            #featuresAudio = np.max(dataWindow)    # calculate features
            #print("Audio features/spectrogram made: ")
            #print("Audio features/max: ", featuresAudio)
            
            predictionStartTime = timeit.default_timer()
            prediction = performInference(model, featuresAudio, input_details, output_details)
            arrayPredictionTime.append(timeit.default_timer() - predictionStartTime)
            print("Prediction made, prediction: ", prediction)

            #displayTimes(arrayQueueTime, arrayWindowTime, arrayFeatureTime, arrayPredictionTime, startTime)
            #print("Total time needed for prediction in ms: ", (timeit.default_timer()-startTime) * 1000)
            #print("Time after prediction is: ", timeit.default_timer())
            arrayQueueTime = []
            arrayWindowTime = []
            arrayFeatureTime = []
            arrayPredictionTime = []

        except queue.Empty:
            emptyCounter = emptyCounter + 1
            #print("empty counter is: ", emptyCounter)

class classAudio:
    def __init__(self, FL, HL):
        print("class audio init")
        #self.device = 16 #Desktop microphone
        #self.device = 14 #Laptop microphone
        self.device = "default"  #RPI
        self.samplerate = 44100
        self.downsamplerate = 16000
        self.downsample = int((self.samplerate/self.downsamplerate))
        self.mapping = 0                                # First channel has index 0
        self.HL = HL

        print("queue init")
        self.q = queue.Queue()                          # Incoming audio is stacked in queue
        print("input stream init")                      #blocksize=int(HL * self.samplerate), samplerate=self.samplerate
        self.stream = sd.InputStream(device=self.device, samplerate=self.samplerate,  channels=1, blocksize=int(HL * self.samplerate), callback=self.callback)    # sound Stream thread, runs to constantly obtain microphone data

        length = int(float(FL) * self.downsamplerate)
        self.dataWindow = np.zeros(length)

        print("window length size: ", length)

        print("thread init")
        self.thread = threading.Thread(
            target=classificationQueue,
            kwargs=dict(
                q=self.q,
                dataWindow=self.dataWindow,
                HL=float(HL)
            ),
        )  # , args =(lambda : stop_threads, ))   # classification thread, constantly classifies incoming data

    def startMicControl(self):
        print("Start microphone control")
        self.stream.start()
        time.sleep(0.1)
        print("Start microphone thread control")
        self.thread.start()

    def stopMicControl(self):
        print("Stop microphone control")
        self.stream.stop()

        self.thread.do_run = False
        # self.thread.join()
        # self.stream.join()

    def callback(self, indata, frames, time, status):
        global callbackTimeTracker
        global resampleTimeTracker

        if status:
            print(status)
        #print("indata completetly is: ", indata)
        #print("indata length is: ", len(indata))
        #print("indata length is with downsample etc: ", len(indata[::self.downsample, self.mapping]))
        #print("indata shape is: ", indata.shape)
        #print("indata with downsample and mapping is: ", indata[::self.downsample, self.mapping])
        #print("indata downsample and mappingshape is: ", indata[::self.downsample, self.mapping].shape)
        #print("indata with squeeze also is: ", np.squeeze(indata[::self.downsample, self.mapping]))
        #print("indata with squeeze shape is: ", np.squeeze(indata[::self.downsample, self.mapping]).shape)
        callbackStartTime = timeit.default_timer()
        if(self.samplerate % self.downsamplerate == 0):
           self.q.put(indata[::self.downsample, self.mapping])
           print("Samplerate + downsamplerate == 0")
           callbackTimeTracker.append(timeit.default_timer() - callbackStartTime)
           return
        resampledData = signal.resample(x=np.squeeze(indata), num=int((self.HL * self.downsamplerate)))
        resampleTimeTracker.append(timeit.default_timer() - callbackStartTime)
        self.q.put(resampledData)
        #print("Resampled data check: ", len(resampledData))
        #self.q.put(indata[::self.downsample, self.mapping])
        #print("Callback current time is: ", timeit.default_timer())
        callbackTimeTracker.append(timeit.default_timer() - callbackStartTime)
        #print("queue length in callback is: ", self.q.qsize())

class Test:
    def __init__(self, parent=None):
        print("Init test 2")
        listDevices
        print(self)
        
        self.FL = None
        self.HL = None
        self.toStart = False

        self.startstopMic()

    def startstopMic(self):
        if self.toStart == True:
            print("start test")
            self.FL = 1
            self.HL = 0.5
            print("Initialize class audio")
            self.Insta = classAudio(self.FL, self.HL)
            print("start Mic control")
            self.Insta.startMicControl()
        else:
            print("stop test")
            self.Insta.stopMicControl()

    def setStatus(self, newStatus):
        self.toStart = newStatus
        print("status set to: ", newStatus)

def startTest():
    print("start test")
    FL = 1
    HL = 0.5
    print("Initialize class audio")
    Insta = classAudio(FL, HL)
    print("start Mic control")
    Insta.startMicControl()


print("ik ben de main, aan het beginnen:")
listDevices()
print("Init test")
testObject = Test
#startTest()
testObject.setStatus(testObject, True)
testObject.startstopMic(testObject)
mainStartTime = time.time()

while True:
    if (time.time() - mainStartTime) > 20:
        print("Stopping")
        testObject.setStatus(testObject, False)
        testObject.startstopMic(testObject)
        break
    #else:
        #print("time is now: ", (time.time() - mainStartTime))
sys.exit()
