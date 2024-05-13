import json, os, shutil, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import seaborn as sns
from hilbert import decode
from PIL import Image
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from mpl_toolkits.axes_grid1 import make_axes_locatable

MainFolder = "E:\\Important\\Skripsi\\Backup\\Skripsi\\Data Test\\Final\\MixedData"
FileListMain = os.listdir(MainFolder)

evaluateLr = False # Toogle to plot learning rate per epoch curve (use only if needed)

# Lists of 2D-transformation methods available
transformMethod = ['Hilbert']

# Colormap Lists
cmapsList = ['viridis']

# Create directories or folder for datasaving purpose
currentDirectory = os.getcwd()
# Main Folder
Gambar = os.path.join(currentDirectory, 'Gambar')
# Folder to save raw plot data
rawPlotPath = os.path.join(Gambar, 'RawPlot')
# Folder to save transformed and colormapped data
transformedPath = os.path.join(Gambar, 'TransformedData')
# Folder to save raw vs transformed spectra graph
spectraGraphPath = os.path.join(Gambar, 'SpectraGraph')
# Folder to save learning curve graph
learningPath = os.path.join(Gambar, 'LearningCurve')
# Folder to save ROC Curve
rocCurvePath = os.path.join(Gambar, 'RocCurve')
# Folder to save Colormapping examples graph
colorExamplePath = os.path.join(Gambar, 'ColorExample')
# Folder to save best model for each transformation-colormapping combination
modelSavePath = os.path.join(Gambar, 'ModelSave')
# Create the Folders
allPaths = [Gambar,
            rawPlotPath,
            transformedPath,
            spectraGraphPath,
            learningPath,
            rocCurvePath,
            colorExamplePath,
            modelSavePath]
for path in allPaths:
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
        
# Function to create a tokenizer for a data
def tokenize_labels(labels):
    # Instantiate the Tokenizer class
    label_tokenizer = Tokenizer(oov_token='<OOV>', lower=False)
    # Fit the tokenizer to the labels
    label_tokenizer.fit_on_texts(labels)
    # Save the word index
    label_word_index = label_tokenizer.word_index
    return label_word_index, label_tokenizer

# Create a tokenizer for the radionuclide part of data
radioLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    radioLabel.append(nameSplit[1])
radioIndex, radioTokenizer = tokenize_labels(radioLabel)

# Create a tokenizer for the atomic number value part of data
atomLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    if len(nameSplit) == 7:
        atomLabel.append(nameSplit[2])
atomIndex, atomTokenizer = tokenize_labels(atomLabel)

# Create a tokenizer for the duration/time part of data
timeLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    timeLabel.append(nameSplit[-3])
timeIndex, timeTokenizer = tokenize_labels(timeLabel)

# Create a tokenizer for the distance part of data
distanceLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    distanceLabel.append(nameSplit[-2])
distanceIndex, distanceTokenizer = tokenize_labels(distanceLabel)

# Create a tokenizer for the iteration part of data
iterationLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    iterationSplit = nameSplit[-1].split('.')
    iterationLabel.append(iterationSplit[0])
iterationIndex, iterationTokenizer = tokenize_labels(iterationLabel)

# To colormap data
def color_maps(data, selectCmap):
    cm = plt.get_cmap(selectCmap)
    mapped_data = cm(data)
    mapped_data = np.uint8(mapped_data * 255)
    return mapped_data

# To find the index of non-oov string according to the used tokenizer
def index_finder(nameSplit, tokenizer):
    tokenizedNameSplit = tokenizer.texts_to_sequences(nameSplit)
    enumerateTokenized = enumerate(tokenizedNameSplit)
    rawIndex = [index for index, value in enumerateTokenized if value!=[1]] # OOV tokens at 1
    return rawIndex[0]

# Model architecture
def create_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(shape=inputs_dim),
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(units=128, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=256, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(units=2, activation='softmax')
  ])
  model.compile(optimizer=RMSprop(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
  return model

# Normalize the spectra data to the range [0, 255]
def normalize_count(x):
    normalized_count = []
    for i in x:
        norm = int((i/max(x)) * 255)
        normalized_count.append(norm)
    return normalized_count

# Preshuffle data and labels to avoid bias
def preshuffle_data_label(data, label):
    random_index = np.random.permutation(data.shape[0])
    preshuffled_data = data[random_index]
    preshuffled_label = [label[i] for i in random_index]
    return preshuffled_data, preshuffled_label

# Hilbert curve type of transformation
def hilbert_curve(data):
    # Check if data is 2D-transformable
    number_of_data = len(data)
    m, n = (int(math.sqrt(number_of_data)), int(math.sqrt(number_of_data)))
    if m*n == number_of_data:
        pass
    else:
        raise ValueError("Sorry, but your data isn't Square 2D-transformable")
    if m == 2:
        raise ValueError("2 x 2 data isn't compatible with Hilbert Curve")
    # Check if the dimension of the resulting 2D matrix is a power of 2
    i = m
    count = 1
    while True:
        if i%2 == 0:
            if i == 2:
                break
            else:
                i = i/2
                count += 1
        else:
            raise ValueError("The dimension of 2D shows that it's not compatible with Hilbert curve")
    # Create placeholder matrix for resulting transformation
    transformed = [[0]*n for _ in range(m)]
    # Obtain hilbert index
    hilbert_index = decode(range(len(data)), 2, count)
    # Place data into placeholder matrix according to hilbert index
    count = 0
    for i,j in hilbert_index:
        transformed[j][i] = data[count]
        count += 1
    return transformed

# Zero Padding
# To pad the data so that it consistently shaped (1, 1024) or (32, 32)
def zero_padding(data):
    shape = (32, 32)
    numData = shape[0]*shape[1]
    if len(data) < numData:
        remainderData = numData - len(data)
        zeroPad = [0 for _ in range(remainderData)]
        data = data + zeroPad
    elif len(data) == numData:
        pass
    else:
        raise ValueError("Number of data exceeds values supported by the algorithm")
    return data

# Learning rate tweaking
def learning_rate_variation(lr):
    return lr * tf.math.exp(-0.1)

# To save transformed and colormapped spectra gamma data
def transformed_data_save(fileName, data, transformedPath, selectTransform, selectCmap):
    img = Image.fromarray(data)
    nameSplit = fileName.split('-')
    iterationSplit = nameSplit[-1].split('.')
    radioTag = nameSplit[index_finder(nameSplit, radioTokenizer)]
    timeTag = nameSplit[index_finder(nameSplit, timeTokenizer)]
    distanceTag = nameSplit[index_finder(nameSplit, distanceTokenizer)]
    iterationTag = iterationSplit[index_finder(iterationSplit, iterationTokenizer)]
    destinationFile = f'{transformedPath}\{radioTag}-{atomIndex}-{timeTag}-{distanceTag}-{selectTransform}-{selectCmap}-{iterationTag}.png'
    if os.path.isfile(destinationFile):
        os.remove(destinationFile)
        img.save(destinationFile)
    else:
        img.save(destinationFile)
        
        # Gama spectra data preprocessing
def data_preprocess(selectCmap, selectTransform):
    totalLabel = []
    placeHolder = np.zeros((1, 32, 32, 4))
    first = True

    for fileName in FileListMain:
        path = MainFolder + f'\{fileName}'
        fileopen = open(path)
        filedata = json.load(fileopen)
        filehist = filedata['histogram']
        
        # Zero padding
        filehist = zero_padding(filehist)

        n = len(filehist)
        for i in range(0, n):
            filehist[i] = i*filehist[i]

        # Normalize the data
        filehist = normalize_count(filehist)
        
        # Apply Hilbert Curve transformation
        data = hilbert_curve(filehist) 

        # Transform the data into numpy array
        data = np.array(data)
        
        # Color mapping
        data = color_maps(data, selectCmap)

        # Save the transformed and colormapped data
        transformed_data_save(fileName, data, transformedPath, selectTransform, selectCmap)
        data = np.expand_dims(data, axis=0)
        
        # if first then stack with placeholder, otherwise stack with the final data
        if first == True:
            totalData = np.vstack((placeHolder, data))
            first = False
        else:
            totalData = np.vstack((totalData, data))
            
        # Adding labels to the data with the help of tokenizer
        if radioTokenizer.texts_to_sequences(fileName.split('-'))[1][0] == radioIndex['co']:
            totalLabel.append([1, 0, 0])
        elif radioTokenizer.texts_to_sequences(fileName.split('-'))[1][0] == radioIndex['cs']:
            totalLabel.append([0, 1, 0])
        elif radioTokenizer.texts_to_sequences(fileName.split('-'))[1][0] == radioIndex['eu']:
            totalLabel.append([0, 0, 1])
        else:
            totalLabel.append([0, 0, 0])
            
        # Close the file
        fileopen.close()
    totalData = totalData[1:] # To erase placeholder data on top
    radioID = [radioIndex['co'], radioIndex['cs'], radioIndex['eu']] # 2 for Co-60, 3 for Cs-137
    return totalData, totalLabel, radioID

# Train the model using K-fold cross falidation
def k_model_train(preshuffled_data, preshuffled_labels, selectTransform, selectCmap):
    # Combining the training and validation data using np.concatenate
    inputs = np.array(preshuffled_data)
    outputs = np.array(preshuffled_labels)
    # Defining K-fold
    kFold = KFold(n_splits=foldAmount, shuffle=False)
    # K-fold commencing
    accPerFold = []
    lossPerFold = []
    historyPerFold = []
    rocPerFold = []
    modelPerFold = []
    foldNumber = 1
    print('--------------------------------------')
    print(f'MODEL WITH {selectTransform} TRANSFORMATION AND {selectCmap} COLORMAPPING')
    for train, test in kFold.split(inputs, outputs):
        model = create_model()
        print('--------------------------------------')
        print(f'Training for fold no-{foldNumber}')
        data = inputs[train]
        labels = outputs[train]
        train_data, val_data = data[:int(splitSize*np.shape(data)[0])], data[int(splitSize*np.shape(data)[0]):]
        train_labels, val_labels = labels[:int(splitSize*np.shape(labels)[0])], labels[int(splitSize*np.shape(labels)[0]):]
        history = model.fit(tf.convert_to_tensor(train_data, dtype=tf.int64),
                            tf.convert_to_tensor(train_labels, dtype=tf.int64),
                            epochs=30,
                            verbose=0,
                            validation_data=(tf.convert_to_tensor(val_data, dtype=tf.int64), 
                                             tf.convert_to_tensor(val_labels, dtype=tf.int64))
                            )
        rocPerRadio = []
        for i in range(len(radioID)):
            y_predict = model.predict(inputs[test], verbose=0)[:, i]
            y_test = outputs[test][:, i]
            fpr, tpr, thresholds = roc_curve(y_test, y_predict)
            rocPerRadio.append((fpr, tpr))
        rocPerFold.append(rocPerRadio)
        evaluation = model.evaluate(inputs[test], outputs[test], verbose=0)
        print(f'Evaluation for fold no-{foldNumber}:')
        print(f'{model.metrics_names[0]} of {evaluation[0]}')
        print(f'{model.metrics_names[1]} of {evaluation[1]}')
        accPerFold.append(evaluation[1] * 100)
        lossPerFold.append(evaluation[0])
        historyPerFold.append((history.history['loss'], 
                            history.history['accuracy'], 
                            history.history['val_loss'], 
                            history.history['val_accuracy']))
        modelPerFold.append(model)
        foldNumber += 1
        # Finding best model from all folds
        if np.argmax(accPerFold) == np.argmin(lossPerFold):
            bestModelIndex = np.argmax(accPerFold)
        else:
            bestModelIndex = np.argmin(lossPerFold)
    print('--------------------------------------')
    print(f'Best model is found to be fold no-{bestModelIndex+1}')
    # Save the best model according to the folds's metrics in .h5 file
    destinationFile = f'{modelSavePath}\{selectTransform}_{selectCmap}.h5'
    if os.path.isfile(destinationFile):
                os.remove(destinationFile)
                modelPerFold[bestModelIndex].save(destinationFile)
    else:
                modelPerFold[bestModelIndex].save(destinationFile)
    # Mean accuracy and mean loss for all folds
    print('--------------------------------------')
    print(f'Mean test accuracy of all folds are found to be: {sum(accPerFold)/len(accPerFold)}')
    print(f'Mean test loss of all folds are found to be: {sum(lossPerFold)/len(lossPerFold)}')
    return rocPerFold, bestModelIndex, historyPerFold

# Plot and save ROC Curve for the best fold
def save_roc_plot(selectTransform, selectCmap, bestModelIndex, radioID, tokenizer1, tokenizer2, rocPerFold, savefolder):
        for id in radioID:
                #plt.plot(rocPerFold[bestModelIndex][id - 2][0], rocPerFold[bestModelIndex][id - 2][1],
                        #label=f'{selectTransform} With {selectCmap} for {tokenizer.sequences_to_texts([[id]])[0]} Data')
                plt.plot(rocPerFold[bestModelIndex][id - 2][0], rocPerFold[bestModelIndex][id - 2][1],
                        label=f'{tokenizer1.sequences_to_texts([[id]])[0].capitalize()}-{tokenizer2.sequences_to_texts([[id]])[0]}')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
        destinationFile = f'{savefolder}\ROC_curve_with_{selectTransform}_and_{selectCmap}.png'
        if os.path.isfile(destinationFile):
                os.remove(destinationFile)
                plt.savefig(destinationFile)
        else:
                plt.savefig(destinationFile)
        plt.close()
        
        # Plot and save learning curve for the best fold
def save_learning_plot(selectTransform, selectCmap, historyPerFold, savefolder):
    f,(ax1,ax2) = plt.subplots(1,2,sharey=False, figsize=(20,5))
    #f.suptitle(f'Learning Curve for Fold no-{bestModelIndex+1} (Best) with {selectTransform} and {selectCmap}', fontsize=20) 
    ax1.plot(list(zip(historyPerFold[bestModelIndex][0], historyPerFold[bestModelIndex][-2])), label=['Loss', 'Validation Loss'])
    ax1.set_xlabel('Epoch')
    ax2.plot(list(zip(historyPerFold[bestModelIndex][1], historyPerFold[bestModelIndex][-1])), label=['Accuracy', 'Validation Accuracy'])
    ax2.set_xlabel('Epoch')
    ax1.legend()
    ax2.legend()
    destinationFile = f'{savefolder}\Learning_curve_with_{selectTransform}_and_{selectCmap}.png'
    if os.path.isfile(destinationFile):
            os.remove(destinationFile)
            f.savefig(destinationFile)
    else:
            f.savefig(destinationFile)
    plt.close()
    
    # Gather filenames of similar distance
def search_distance_data(preferDistance):
    filesNeeded = []
    for filename in FileListMain:
        nameSplit = filename.split('-')
        distanceTag = nameSplit[index_finder(nameSplit, distanceTokenizer)]
        if (preferDistance == distanceTag):
            filesNeeded.append(filename)
    if len(filesNeeded) == 0:
        raise ValueError("Wrong inquiry parameters, please try again")
    return filesNeeded

# Gather filenames of similar time
def search_time_data(preferTime):
    filesNeeded = []
    for filename in FileListMain:
        nameSplit = filename.split('-')
        timeTag = nameSplit[index_finder(nameSplit, timeTokenizer)]
        if (preferTime == timeTag):
            filesNeeded.append(filename)
    if len(filesNeeded) == 0:
        raise ValueError("Wrong inquiry parameters, please try again")
    return filesNeeded

# Define the raw_data_save() function
def raw_data_save(fileName, data, radioTag, atomTag, timeTag, distanceTag, iterationTag, rawPlotPath):
    plt.figure(figsize=(10, 6))
    plt.plot(data, color='darkcyan', linestyle='-', linewidth=1, markersize=1)
    plt.xlabel('Energy Bin/Channel')
    plt.ylabel('Normalized Counts')
    plt.title(f'{radioTag} {atomTag} 1K {timeTag} {distanceTag} {iterationTag}')
    
    save_path = os.path.join(rawPlotPath, f'{fileName.split(".json")[0]}.png')
    plt.savefig(save_path)
    plt.close()

# # Call the raw_data_save() function in the process_and_save_data() function
# def process_and_save_data(fileName, path, radioTag, atomTag, timeTag, distanceTag, iterationTag, rawPlotPath):
#     with open(path) as file:
#         data = json.load(file)['histogram']
#         data = zero_padding(data)
#         data = np.array(data)
        
#     raw_data_save(fileName, data, radioTag, atomTag, timeTag, distanceTag, iterationTag, rawPlotPath)
    
for fileName in FileListMain:
    with open(path) as file:
        data = json.load(file)['histogram']
        data = zero_padding(data)
        data = np.array(data)
        nameSplit = fileName.split("-")
        radioTag = nameSplit[1]
        atomTag = nameSplit[2]
        timeTag = nameSplit[index_finder(nameSplit, timeTokenizer)]
        distanceTag = nameSplit[index_finder(nameSplit, distanceTokenizer)]
        iterationSplit = nameSplit[-1].split('.')
        iterationTag = iterationSplit[index_finder(iterationSplit, iterationTokenizer)]
        
        if radioTag in radioIndex:  # Check if radioTag is a valid radioIndex
            # Process data and save plot        
            if radioTag == 'BckGnd':
                raw_data_save(fileName, data, radioTag, atomTag, timeTag, distanceTag, iterationTag, rawPlotPath)
            elif radioTag == 'Co':
                raw_data_save(fileName, data, radioTag, atomTag, timeTag, distanceTag, iterationTag, rawPlotPath)
            elif radioTag == 'Cs':
                raw_data_save(fileName, data, radioTag, atomTag, timeTag, distanceTag, iterationTag, rawPlotPath)
            elif radioTag == 'Eu':
                raw_data_save(fileName, data, radioTag, atomTag, timeTag, distanceTag, iterationTag, rawPlotPath)
            else:
                raise ValueError("Unknown radioTag")
        else:
            print(f"Ignore file '{fileName}' as it has invalid radioTag '{radioTag}'.")