#Perform the following commands before starting, otherwise saving to GDrive will not work
#pip install httplib2==0.15.0
#pip install google-api-python-client==1.7.11
#Additionally all the modules list below, need to be installed in Python

import os, sys, random, time, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

random.seed(69)
skipConvert = False #Skips converting of opus to wav for debugging
chunkSize = 10 ** 3 #Size of the chunks to use, no need to change

keywords = ["red", "blue"] #Keywords to convert
mswcPath = Path("E:\\Datasets Hankamp (Non-blowing)\\MSWC\\English\\") #Path of the MSWC dataset, use double \ for all Paths
wavPath = Path("E:\\Datasets Hankamp (Non-blowing)\\MSWC\\English\\en\\wavs\\") #Destination path for the wav files
unknownPath = Path("E:\\Datasets Hankamp (Non-blowing)\\MSWC\\NL\\audio\\nl\clips\\") #Path to generate unknowns from
zipFileTitle = "MSWC_keywords_wavs"
googleDriveFolderName = "MSWC_dataset_wavs" #GoogleDrive folder name, where to save all the keyword wav files
unknownNrPerWord = 10 #Amount of samples per random word that will be included in the unknown class

def check_directory(set, word): #Check if a directory exists for the correct split and word, if not make one
    wordDirectory = (wavPath / set / word) #Directory where the current word should be saved
    if os.path.isdir(wordDirectory) != True: os.makedirs(wordDirectory) #Makes the directory tree if it doesn't exists
    return wordDirectory

def opusToWav(source, dest):
    cmd = ["opusdec.exe", "--rate", "16000", source, dest] #Use opusdec.exe from opustools to convert opus file to a wav
    subprocess.run(cmd)
    return

def mswcToWav(row): #Coverts and saves the opus keywords to a wav file
    source = (mswcPath / "en/clips" / row["LINK"]) #File path of the opus file

    dest = check_directory(row["SET"], row["WORD"])
    dest = dest / ((row["LINK"].split("/")[-1]).split(".")[0] + ".wav") #Destination file path, use same name but without the .opus extention

    if os.path.isfile(dest) == True: #Only convert files once
        print("WAV already exists, skipping...")
        return
    opusToWav(source, dest)
    return

def checkAverageKeywordAmount():
    averageTrain = []
    averageValTest = []
    for i in keywords:
        keywordAmountTotal = len(os.listdir((mswcPath / "en/clips" / i)))
        print(i, "has,", keywordAmountTotal, "samples in total")

        keywordAmountTrain = len(os.listdir((wavPath / "TRAIN" / i)))
        keywordAmountValTest = keywordAmountTotal - keywordAmountTrain
        print(i, "has,", keywordAmountTrain, "samples for training")
        print(i, "has,", keywordAmountValTest, "samples for validation and testing")
        averageTrain.append(keywordAmountTrain)
        averageValTest.append(keywordAmountValTest)
        print()

    averageTrain = int(np.average(averageTrain))
    averageValTest = int(np.average(averageValTest))
    ratioTrain = round((averageTrain / (averageTrain + averageValTest)), 1)
    print("Average amount of train samples:", averageTrain, " Average number of test+val samples:", averageValTest)
    print("Ratio: ", ratioTrain)
    print((averageTrain + averageValTest))
    return averageTrain, averageValTest, ratioTrain

def saveToGDrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()           
    drive = GoogleDrive(gauth)
    folder = drive.CreateFile({'title' : googleDriveFolderName, 'mimeType' : 'application/vnd.google-apps.folder'})
    folder.Upload()
    print("Folder made in GDrive, ID: ", folder.get('id'))

    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file1 in file_list:       
        print ('title: %s, id: %s' % (file1['title'], file1['id']))

    zipFilePath = shutil.make_archive((mswcPath / zipFileTitle), 'zip', root_dir=wavPath)
    print(zipFilePath)
    zipFile = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder.get('id')}],
                                "title": (zipFileTitle + ".zip")})
    zipFile.SetContentFile(zipFilePath)
    zipFile.Upload()
    return

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