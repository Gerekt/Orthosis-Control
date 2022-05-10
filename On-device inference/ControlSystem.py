commands = ['blue', 'red', '_unknown_'] #Network is trained with commands in alphabetical order, enter the commands that were used during training here

import os, sys
import pwd
import numpy as np
import sounddevice as sd
import time
import timeit
import queue
import threading
from scipy import signal
import tensorflow as tf
from numpy.lib.shape_base import expand_dims
from gpiozero import Button, Motor

from wavToSpectrogram import get_scipy_spectrogram

modelName = "modelV5.tflite" #Name and path of model that will be used, program currently only accepts tflite models(Can be changed easily)

callbackTimeTracker = []
resampleTimeTracker = []
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Ignores message that a certain instruction set is not used.

micActivation = Button(5)
orthosisMotor = Motor(6, 12)
activationTime  = 15
lastActivationTime = 0

def micActivator():
    global lastActivationTime
    print()
    print("Mic activation pressed!")
    lastActivationTime = timeit.default_timer()
    print(lastActivationTime)

def moveOrthosis(prediction):
    global orthosisMotor
    if prediction == "_unknown_":
        print("Unknown, not moving")
        return
    elif prediction == "red":
        print("red, moving backward")
        orthosisMotor.backward()
    elif prediction == "blue":
        print("blue, moving forward")
        orthosisMotor.forward()

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
    global modelName
    if os.path.isfile(modelName) != True:
        print("Model path not found")
        return False
    else:
        Interpreter = tf.lite.Interpreter
        load_delegate = tf.lite.experimental.load_delegate

        model = Interpreter(model_path=modelName) #Looks for tfllite model in this folder
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
    global lastActivationTime
    global activationTime
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
            data = q.get(True, 1) # Wait until full hoplength of samples is ready in queue, waits no longer than 1 sec
            arrayQueueTime.append(timeit.default_timer() - startTime)
            windowStartTime = timeit.default_timer()
            shift = len(data)
            dataWindow = np.roll(dataWindow, -shift)            # dataWindow is total window length over which the features are calculated
            dataWindow[-shift:] = data
            arrayWindowTime.append(timeit.default_timer() - windowStartTime)                                  # new incoming raw microphone data is added to window

            if timeit.default_timer() <= lastActivationTime + activationTime:
                featureStartTime = timeit.default_timer()
                featuresAudio = get_scipy_spectrogram(dataWindow)
                arrayFeatureTime.append(timeit.default_timer() - featureStartTime)
                predictionStartTime = timeit.default_timer()
                prediction = performInference(model, featuresAudio, input_details, output_details)
                arrayPredictionTime.append(timeit.default_timer() - predictionStartTime)
                print("Prediction made, prediction: ", prediction)
                moveOrthosis(prediction)
            else:
                print("Mic not activated, no prediction made")
            #displayTimes(arrayQueueTime, arrayWindowTime, arrayFeatureTime, arrayPredictionTime, startTime)
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
        self.device = "default"  #Device which will be used for the microphone input
        self.samplerate = 44100 #Native samplerate of the microphone/ADC, 16kHz would be preferable but was unable to get this to work
        self.downsamplerate = 16000 #Samplerate which will be used for spectrograms, needs to be the same as was used during training.
        self.downsample = int((self.samplerate/self.downsamplerate)) #Downsample ratio
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
            self.FL = 1 #Initialization of the Frame and hoplength. Model V5.0 is trained using 1 secocond framelength. Hoplength may be changed regardless of framelength during training
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

print("ik ben de main, aan het beginnen:")
listDevices()
print("Init test")
testObject = Test
testObject.setStatus(testObject, True)
testObject.startstopMic(testObject)
mainStartTime = time.time()

while True:
    if (time.time() - mainStartTime) > 20: #Currently the program shutsdown after 20 seconds, change this look for unlimited runtime
        print("Stopping")
        testObject.setStatus(testObject, False)
        testObject.startstopMic(testObject)
        break
    #else:
        #print("time is now: ", (time.time() - mainStartTime))
sys.exit()
