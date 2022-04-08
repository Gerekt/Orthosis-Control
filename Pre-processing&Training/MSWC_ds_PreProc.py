#Perform the following commands before starting, otherwise saving to GDrive will not work
#pip install httplib2==0.15.0
#pip install google-api-python-client==1.7.11
#Additionally all the modules list below, need to be installed in Python

#Load libraries
import os, sys, random, time, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from mswcHelperFunctions import check_directory, opusToWav, mswcToWav, checkAverageKeywordAmount, saveToGDrive

random.seed(69) #Set the random seed so that results can be repeated
skipConvert = False #Skips converting of opus to wav for debugging, must be set to False if an new dataset is to be created
chunkSize = 10 ** 3 #Size of the chunks to use, no need to change
unknownNrPerWord = 10 #Amount of samples per random word that will be included in the unknown class
keywords = ["red", "blue"] #Keywords to convert, must be present in the selected dataset

mswcPath = Path("E:\\Datasets Hankamp (Non-blowing)\\MSWC\\English\\") #Path of the MSWC dataset, use double \ for all Paths
wavPath = Path("E:\\Datasets Hankamp (Non-blowing)\\MSWC\\English\\en\\wavs\\") #Destination path for the wav files
unknownPath = Path("E:\\Datasets Hankamp (Non-blowing)\\MSWC\\NL\\audio\\nl\clips\\") #Path to generate unknowns from
zipFileTitle = "MSWC_keywords_wavs" #Name of the output zipfile containing the dataset that is made
googleDriveFolderName = "MSWC_dataset_wavs" #GoogleDrive folder name, where to save all the keyword wav files



if skipConvert == False:
    with pd.read_csv((mswcPath / "en_splits.csv"), chunksize=chunkSize, usecols=["SET", "LINK", "WORD"]) as reader:
        for chunk in reader: #Read CSV files in chunks due to the large size of the file
            for index, row in chunk.iterrows(): #Iterate through each row of the chunk
                if row["WORD"] in keywords: #Only convert the selected keywords
                    mswcToWav(row)
    print("Done coverting and saving keywords")

print("Making unknown set")
averageTrain, averageValTest, ratioTrain = checkAverageKeywordAmount()
for files in range(0, (averageTrain + averageValTest), unknownNrPerWord):
    randomDir = random.choice(os.listdir(unknownPath))
    while len(os.listdir(unknownPath / randomDir)) < unknownNrPerWord:
        print("Directory smaller than the minimum of: ", unknownNrPerWord)
        randomDir = random.choice(os.listdir(unknownPath))
    #print("Random directory:", randomDir, "Amount of files: ", len(os.listdir(unknownPath / randomDir)))

    filesSource = []
    for i in range(unknownNrPerWord):
        randomFile = random.choice(os.listdir(unknownPath / randomDir))
        while randomFile in filesSource:
            print("Random file already presentfile: ", randomFile, " Grabbing new one...")
            randomFile = random.choice(os.listdir(unknownPath / randomDir))
        filesSource.append(randomFile)

    if os.path.isdir(wavPath / "TRAIN/unknown") != True: os.makedirs(wavPath / "TRAIN/unknown")
    if os.path.isdir(wavPath / "TEST/unknown") != True: os.makedirs(wavPath / "TEST/unknown")
    if os.path.isdir(wavPath / "DEV/unknown") != True: os.makedirs(wavPath / "DEV/unknown")

    nrTrainFiles = int(ratioTrain * unknownNrPerWord)
    nrTestFiles = int((unknownNrPerWord-nrTrainFiles + 1)/2) - 1
    for unknownTrain in filesSource[:nrTrainFiles]:
        unknownTrain = randomDir + "/"+ unknownTrain
        dest = (wavPath / "TRAIN/unknown" / ((unknownTrain.split("/")[-1]).split(".")[0] + ".wav"))
        print(dest)
        print((unknownPath / unknownTrain))
        opusToWav((unknownPath / unknownTrain), dest)
    for unknownDevTest in filesSource[nrTrainFiles:(nrTrainFiles + nrTestFiles)+1]:
        dest = (wavPath / "DEV/unknown" / ((unknownTrain.split("/")[-1]).split(".")[0] + ".wav"))
        print(dest)
        opusToWav((unknownPath / unknownTrain), dest)
    for unknownDevTest in filesSource[(nrTrainFiles + nrTestFiles)+1:]:
        dest = (wavPath / "TEST/unknown" / ((unknownTrain.split("/")[-1]).split(".")[0] + ".wav"))
        print(dest)
        opusToWav((unknownPath / unknownTrain), dest)

saveToGDrive()