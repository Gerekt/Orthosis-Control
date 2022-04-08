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

def checkAverageKeywordAmount(): #Checks for the average amount of keywords that is present in the dataset
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

def saveToGDrive(): #Saves the files to Google Drive
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
