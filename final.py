# %% [markdown]
# # Import Library

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
import json, os, shutil, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from hilbert import decode
from PIL import Image
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
from sklearn.metrics import confusion_matrix


# %% [markdown]
# # Open The Main Data Folder

# %%
MainFolder = "MixedData"
FileListMain = os.listdir(MainFolder)

evaluateLr = False # Toogle to plot learning rate per epoch curve (use only if needed)
useColormap = True # Toogle to use colormapping

# Lists of transformation methods available
transformMethod = ['Normal', 'Upscale', 'Descale']
BgMethod = ['Normal', 'NonZero']
RawMethod = ['Normal', 'Normalized']
ModelMethod = ['ANN', 'CNN']
# Amount of folds for k-fold cross validation
foldAmount = 5

# Size of training data for train/val split
splitSize = 0.8

# %% [markdown]
# # Make the Folder

# %%
# Create directories or folder for datasaving purpose
currentDirectory = os.getcwd()
# Main Folder
Gambar = os.path.join(currentDirectory, 'Gambar')
# Folder to save raw plot data
rawPlotPath = os.path.join(Gambar, 'RawPlot')
# Folder to save normalized raw data
normalizedRawPath = os.path.join(Gambar, 'NormalizedRaw')
# Folder to save background data
backgroundPath = os.path.join(Gambar, 'Background')
# Folder to save removed background data
removedBackgroundPath = os.path.join(Gambar, 'RemovedBackground')
# Folder to save removed Background data non zero normalized
removedBackgroundPathNonZero = os.path.join(Gambar, 'RemovedBackgroundNonZero')
# Folder to save transformed and colormapped data
transformedPath = os.path.join(Gambar, 'TransformedData')
# Folder to save learning curve graph
evaluatePath = os.path.join(Gambar, 'Evaluate')
# Folder to save best model for each transformation-colormapping combination
modelSavePath = os.path.join(Gambar, 'ModelSave')
# Create the Folders
allPaths = [Gambar,
            rawPlotPath,
            normalizedRawPath,
            backgroundPath,
            removedBackgroundPath,
            removedBackgroundPathNonZero,
            transformedPath,
            evaluatePath,
            modelSavePath]
for path in allPaths:
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

# %% [markdown]
# # Tokenizer

# %%
# Function to create a tokenizer for a data
def tokenize_labels(labels):
    # Instantiate the Tokenizer class
    label_tokenizer = Tokenizer(oov_token='<OOV>', lower=False)
    # Fit the tokenizer to the labels
    label_tokenizer.fit_on_texts(labels)
    # Save the word index
    label_word_index = label_tokenizer.word_index
    return label_word_index, label_tokenizer

# %%
# Create a tokenizer for the radionuclide part of data
radioLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    radioLabel.append(nameSplit[1])
radioIndex, radioTokenizer = tokenize_labels(radioLabel)
print(radioIndex)

# %%
# Create a tokenizer for the atomic number value part of data
atomLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    if len(nameSplit) == 7:
        atomLabel.append(nameSplit[2])
atomIndex, atomTokenizer = tokenize_labels(atomLabel)
print(atomIndex)

# %%
# Create a tokenizer for the duration/time part of data
timeLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    timeLabel.append(nameSplit[-3])
timeIndex, timeTokenizer = tokenize_labels(timeLabel)
print(timeIndex)

# %%
# Create a tokenizer for the distance part of data
distanceLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    distanceLabel.append(nameSplit[-2])
distanceIndex, distanceTokenizer = tokenize_labels(distanceLabel)
print(distanceIndex)

# %%
# Create a tokenizer for the iteration part of data
iterationLabel = []
for fileName in FileListMain:
    nameSplit = fileName.split("-")
    iterationSplit = nameSplit[-1].split('.')
    iterationLabel.append(iterationSplit[0])
iterationIndex, iterationTokenizer = tokenize_labels(iterationLabel)
print(iterationIndex)

# %% [markdown]
# # Define All the Function

# %%
# Upscale
def upscale(data, size=(64, 64)):
    data = resize(data, (size[0], size[1], data.shape[2]), mode='constant')
    return data  # Return the upscaled data

# Descale
def descale(data, size=(16, 16)):
    data = resize(data, (size[0], size[1], data.shape[2]), mode='constant')
    return data  # Return the descaled data

# %%
# To colormap data
def color_maps(data):
    cm = plt.get_cmap('viridis')
    mapped_data = cm(data)
    mapped_data = np.uint8(mapped_data * 255)
    return mapped_data

# %%
# To find the index of non-oov string according to the used tokenizer
def index_finder(nameSplit, tokenizer):
    tokenizedNameSplit = tokenizer.texts_to_sequences(nameSplit)
    enumerateTokenized = enumerate(tokenizedNameSplit)
    rawIndex = [index for index, value in enumerateTokenized if value!=[1]] # OOV tokens at 1
    return rawIndex[0]

# %%
# Normalize the spectra data to the range [0, 255]
def normalize_count(x):
    normalized_count = []
    for i in x:
        norm = int((i/max(x)) * 255)
        normalized_count.append(norm)
    return normalized_count

# %%
# Preshuffle data and labels to avoid bias
def preshuffle_data_label(data, label):
    random_index = np.random.permutation(data.shape[0])
    preshuffled_data = data[random_index]
    preshuffled_label = [label[i] for i in random_index]
    return preshuffled_data, preshuffled_label

# %%
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

# %%
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

# %%
# Learning rate tweaking
def learning_rate_variation(lr):
    return lr * tf.math.exp(-0.1)

# %%
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

# %%
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

# %% [markdown]
# # Understanding The Data

# %%
# Define the raw_data_plot() function
def raw_data_plot(selectMethod):
    for fileName in FileListMain:
        nameSplit = fileName.split("-")
        radioTag = nameSplit[1]
        atomTag = nameSplit[2]
        timeTag = nameSplit[index_finder(nameSplit, timeTokenizer)]
        distanceTag = nameSplit[index_finder(nameSplit, distanceTokenizer)]
        iterationSplit = nameSplit[-1].split('.')
        iterationTag = iterationSplit[index_finder(iterationSplit, iterationTokenizer)]

        if radioTag not in ['BckGnd', 'Co', 'Cs', 'Eu']:
            continue
            
        path = MainFolder + f'\{fileName}'
        data = open(path)
        data = json.load(data)
        data = data['histogram']
        data = zero_padding(data)
        
        if selectMethod == 'Normal':
            data = np.array(data)
            plt.title(f'{radioTag} {atomTag} 1K {timeTag} {distanceTag} {iterationTag}')
            save_path = os.path.join(rawPlotPath, f'{fileName.split(".json")[0]}.png')
            plt.ylabel('Counts')

        else:
            data = np.array(data)
            data = normalize_count(data)
            plt.title(f'{radioTag} {atomTag} 1K {timeTag} {distanceTag} {iterationTag}')
            save_path = os.path.join(normalizedRawPath, f'{fileName.split(".json")[0]}.png')
            plt.ylabel('Normalized Counts')

        plt.figure(figsize=(10, 6))
        plt.plot(data, color='darkcyan', linewidth=1, markersize=1)
        plt.xlabel('Energy Bin/Channel')
        plt.title(f'{radioTag} {atomTag} {timeTag} {distanceTag} {iterationTag}')
        
        # Check if the file already exists
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove the existing file
        plt.savefig(save_path)
        plt.close()
    plt.close()

# %%
for RawMode in RawMethod:
    raw_data_plot(RawMode)

# %% [markdown]
# # Background Data Preprocessing

# %%
# Tanpa Normalisasi
first = True
count = 0
totalBack = None  # Initialize totalBack as None
for filename in FileListMain:
    path = MainFolder + f'\{filename}'
    nameSplit = filename.split('-')
    if nameSplit[index_finder(nameSplit, radioTokenizer)] == 'BckGnd':
        count += 1
        with open(path) as fileopen:
            filedata = json.load(fileopen)
            filehist = filedata['histogram']
            filehist = zero_padding(filehist)
            filehist = np.array(filehist)
        if first == True:
            totalBack = filehist
            first = False
        else:
            totalBack = totalBack + filehist

if count != 0:
    totalBack = totalBack/count  # Compute the average histogram

    # Plot the average histogram
    plt.figure(figsize=(10, 6))
    plt.plot(totalBack, color='darkcyan', linewidth=1, markersize=1)
    plt.xlabel('Energi Bin/Channel')
    plt.ylabel('Counts')
    plt.title('Rata-Rata Background Spectrum')
else:
    print("No background files found.")

save_path = os.path.join(backgroundPath, 'Average_Background_Spectrum.png')
# Check if the file already exists
if os.path.exists(save_path):
    os.remove(save_path)  # Remove the existing file
plt.savefig(save_path)
plt.close()

# %%
# Normalisasi sebelum Rata-rata
first = True
count = 0
totalBack = None  # Initialize totalBack as None
for filename in FileListMain:
    path = MainFolder + f'\{filename}'
    nameSplit = filename.split('-')
    if nameSplit[index_finder(nameSplit, radioTokenizer)] == 'BckGnd':
        count += 1
        with open(path) as fileopen:
            filedata = json.load(fileopen)
            filehist = filedata['histogram']
            filehist = zero_padding(filehist)
            filehist = normalize_count(filehist)
            filehist = np.array(filehist)
        if first == True:
            totalBacknorm = filehist
            first = False
        else:
            totalBacknorm = totalBacknorm + filehist

if count != 0:
    totalBacknorm = totalBacknorm/count  # Compute the average histogram

    # Plot the average histogram
    plt.figure(figsize=(10, 6))
    plt.plot(totalBacknorm, color='darkcyan', linewidth=1, markersize=1)
    plt.xlabel('Energi Bin/Channel')
    plt.ylabel('Counts')
    plt.title('Rata-Rata Background Spectrum')
else:
    print("No background files found.")
    
save_path = os.path.join(backgroundPath, 'Normalized_Average_Background_Spectrum.png')
# Check if the file already exists
if os.path.exists(save_path):
    os.remove(save_path)  # Remove the existing file
plt.savefig(save_path)
plt.close()

# %% [markdown]
# # Removing Background Data

# %%
# Define remove_background() function
def remove_background(selectMethod, totalBacknorm):
    for fileName in FileListMain:
        nameSplit = fileName.split("-")
        radioTag = nameSplit[index_finder(nameSplit, radioTokenizer)]
        timeTag = nameSplit[index_finder(nameSplit, timeTokenizer)]
        distanceTag = nameSplit[index_finder(nameSplit, distanceTokenizer)]
        iterationSplit = nameSplit[-1].split('.')
        iterationTag = iterationSplit[index_finder(iterationSplit, iterationTokenizer)]
        
        path = MainFolder + f'\{fileName}'
        if radioTag not in ['Co', 'Cs', 'Eu']:
            continue
        
        data = open(path)
        data = json.load(data)
        data = data['histogram']
        data = zero_padding(data)
        data = normalize_count(data)
        data = np.array(data)
        data = data - totalBacknorm

        if selectMethod == 'Normal':
            save_path = os.path.join(removedBackgroundPath, f'{fileName.split(".json")[0]}.png')

        else:
            data = [i if i > 0 else 0 for i in data]
            save_path = os.path.join(removedBackgroundPathNonZero, f'{fileName.split(".json")[0]}.png')
            
        plt.figure(figsize=(10, 6))
        plt.title(f'Removed Background {radioTag} {timeTag} {distanceTag}-{iterationTag}')
        plt.plot(data, color='darkcyan', linewidth=1, markersize=1)
        plt.xlabel('Energy Bin/Channel')
        plt.ylabel('Counts')
        
        # Check if the file already exists
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove the existing file
        plt.savefig(save_path)
        plt.close()
    plt.close()

# %%
for method in BgMethod:
    remove_background(method, totalBacknorm)

# %% [markdown]
# # Save Transformation Data

# %%
# To save transformed and colormapped spectra gamma data
def transformed_data_save(fileName, selectTransform, data, transformedPath, radioTag, atomTag, timeTag, distanceTag, iterationTag): 
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='viridis')
    plt.xlabel('Energy Bin/Channel')
    plt.ylabel('Normalized Counts')
    
    save_path = os.path.join(transformedPath, f'{selectTransform}-{fileName.split(".json")[0]}.png')
    
    plt.title(f'{selectTransform} Transformed {radioTag} {atomTag} 1K {timeTag} {distanceTag} {iterationTag}')

    # Check if the file already exists
    if os.path.exists(save_path):
        os.remove(save_path)  # Remove the existing file
    plt.savefig(save_path)
    plt.close()

# %%
# Gama spectra data preprocessing
def data_preprocess_image(selectTransform):       
    for fileName in FileListMain:
        nameSplit = fileName.split('-')
        iterationSplit = nameSplit[-1].split('.')
        radioTag = nameSplit[index_finder(nameSplit, radioTokenizer)]
        timeTag = nameSplit[index_finder(nameSplit, timeTokenizer)]
        distanceTag = nameSplit[index_finder(nameSplit, distanceTokenizer)]
        iterationTag = iterationSplit[index_finder(iterationSplit, iterationTokenizer)]
        
        path = MainFolder + f'\{fileName}'
        if radioTag not in ['Co', 'Cs', 'Eu']:
            continue
            
        fileopen = open(path)
        filedata = json.load(fileopen)
        data = filedata['histogram']
        data = zero_padding(data)              
        data = normalize_count(data) - totalBacknorm
        data = hilbert_curve(data)
        data = color_maps(data)
        data = np.array(data)
        
                # Use Upscale/Descale method
        if selectTransform == 'Upscale':
            data = upscale(data)
        elif selectTransform == 'Descale':
            data = descale(data)
        else:
            pass
        
        # Save the transformed and colormapped data
        
        if selectTransform == 'Upscale':
            transformed_data_save(fileName, selectTransform, data, transformedPath, radioTag, radioTag, timeTag, distanceTag, iterationTag)
        elif selectTransform == 'Descale':
            transformed_data_save(fileName, selectTransform, data, transformedPath, radioTag, radioTag, timeTag, distanceTag, iterationTag)
        elif selectTransform == 'Normal':
            transformed_data_save(fileName, selectTransform, data, transformedPath, radioTag, radioTag, timeTag, distanceTag, iterationTag)
        else:
            pass

# %%
# Apply Transformation and Colormapping
for transform in transformMethod:
    data_preprocess_image(transform)

# %% [markdown]
# # Modeling and Training

# %%
# Gama spectra data preprocessing
def data_preprocess(selectTransform):
    totalLabel = []
    if selectTransform == 'Upscale':
        placeHolder = np.zeros((1, 64, 64, 4))
    elif selectTransform == 'Descale':
        placeHolder = np.zeros((1, 16, 16, 4))
    else:
        placeHolder= np.zeros((1, 32, 32, 4))
    first = True
        
    for fileName in FileListMain:
        nameSplit = fileName.split('-')
        iterationSplit = nameSplit[-1].split('.')
        radioTag = nameSplit[index_finder(nameSplit, radioTokenizer)]
        timeTag = nameSplit[index_finder(nameSplit, timeTokenizer)]
        distanceTag = nameSplit[index_finder(nameSplit, distanceTokenizer)]
        iterationTag = iterationSplit[index_finder(iterationSplit, iterationTokenizer)]
        
        path = MainFolder + f'\{fileName}'
        if radioTag not in ['Co', 'Cs', 'Eu']:
            continue
            
        fileopen = open(path)
        filedata = json.load(fileopen)
        data = filedata['histogram']
        data = zero_padding(data)              
        data = normalize_count(data) - totalBacknorm
        data = hilbert_curve(data)
        data = color_maps(data)
        data = np.array(data)
        
                # Use Upscale/Descale method
        if selectTransform == 'Upscale':
            data = upscale(data)
        elif selectTransform == 'Descale':
            data = descale(data)
        else:
            pass
        
        # Save the transformed and colormapped data
        data = np.expand_dims(data, axis=0)
        
        # if first then stack with placeholder, otherwise stack with the final data       
        if first == True:
            totalData = np.vstack((placeHolder, data))
            first = False
        else:
            totalData = np.vstack((totalData, data))
        
        if radioTag in radioIndex:  # Check if radioTag is a valid radioIndex
        # Process data and save plot
            if radioTag == 'Co':
                totalLabel.append([1, 0, 0])
            elif radioTag == 'Cs':
                totalLabel.append([0, 1, 0])
            elif radioTag == 'Eu':
                totalLabel.append([0, 0, 1])
            else:
                pass
        else:
            print(f"Ignore file '{fileName}' as it has invalid radioTag '{radioTag}'.")
        fileopen.close()
    if useColormap == True:
        pass
    else:
        totalData = np.expand_dims(totalData, axis=-1)
        
        # Close the file
    totalData = totalData[1:] # To erase placeholder data on top
    radioID = [radioIndex['Co'], radioIndex['Cs'], radioIndex['Eu']] # 3 for Co-60, 4 for Cs-137, 5 for Eu-152
    return totalData, totalLabel, radioID

# %%
# CNN Model architecture
def create_cnn_model(input_dims):
  model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(units=128, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=256, activation='relu'),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(units=3, activation='softmax')
  ])
  model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
  return model

# %%
# ANN Model architecture
def create_ann_model(input_dims):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_dims),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# %%
def k_model_train(preshuffled_data, preshuffled_label, input_dims, selectTransform, selectModel, radioID):
    # Convert input data and labels to lists of indices
    indices = np.arange(len(preshuffled_data))
    np.random.shuffle(indices)

    inputs = np.array(preshuffled_data)[indices]
    outputs = np.array(preshuffled_label)[indices]

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
    print(f'{selectModel} MODEL WITH {selectTransform} TRANSFORMATION')

    for train, test in kFold.split(inputs):
        if selectModel == 'CNN':
            model_creator = create_cnn_model
        elif selectModel == 'ANN':
            model_creator = create_ann_model
        else:
            raise ValueError("Invalid model type. Use 'CNN' or 'ANN'.")

        model = model_creator(input_dims)
        
        print('--------------------------------------')
        print(f'Training for fold no-{foldNumber}')
        
        train_data, val_data = inputs[train], inputs[test]
        train_labels, val_labels = outputs[train], outputs[test]

        history = model.fit(train_data, train_labels,
                            epochs=30,
                            verbose=0,
                            validation_data=(val_data, val_labels))
        
        rocPerRadio = []
        for i in range(len(radioID)):
            y_predict = model.predict(val_data, verbose=0)[:, i]
            y_test = val_labels[:, i]
            fpr, tpr, thresholds = roc_curve(y_test, y_predict)
            rocPerRadio.append((fpr, tpr))
        rocPerFold.append(rocPerRadio)
        
        evaluation = model.evaluate(val_data, val_labels, verbose=0)
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
    destinationFile = f'{modelSavePath}\{selectTransform}.h5'
    if os.path.isfile(destinationFile):
        os.remove(destinationFile)
        modelPerFold[bestModelIndex].save(destinationFile)
    else:
        modelPerFold[bestModelIndex].save(destinationFile)
    
    # Mean accuracy and mean loss for all folds
    print('--------------------------------------')
    print(f'Mean test accuracy of all folds are found to be: {sum(accPerFold)/len(accPerFold)}')
    print(f'Mean test loss of all folds are found to be: {sum(lossPerFold)/len(lossPerFold)}')
    
    return rocPerFold, bestModelIndex, historyPerFold, modelPerFold


# %%
def save_roc_plot(selectTransform, selectMethod, bestModelIndex, radioID, tokenizer1, tokenizer2, rocPerFold, savefolder):
    if bestModelIndex >= len(rocPerFold):
        return

    for id in radioID:
        index = id - 3
        if index < 0 or index >= len(rocPerFold[bestModelIndex]):
            print(f'No data available for Radio ID: {id} at index: {index}')
            continue
        
        fpr, tpr = rocPerFold[bestModelIndex][index]

        if fpr.size > 0 and tpr.size > 0:  # Use .size to check for non-empty numpy arrays
            label_text = f'{tokenizer1.sequences_to_texts([[id]])[0].capitalize()}-{tokenizer2.sequences_to_texts([[id-1]])[0]}'
            plt.plot(fpr, tpr, label=label_text)

    plt.title(f'{selectMethod} Model of ROC curve with {selectTransform}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    destinationFile = os.path.join(savefolder, f'{selectMethod}_Model_of_ROC_curve_with_{selectTransform}.png')
    if os.path.isfile(destinationFile):
        os.remove(destinationFile)
    plt.savefig(destinationFile)
    plt.close()

# %%
# Plot and save learning curve for the best fold
def save_learning_plot(selectTransform, bestModelIndex, selectMethod, historyPerFold, savefolder):
    f,(ax1,ax2) = plt.subplots(1,2,sharey=False, figsize=(20,5))
    f.suptitle(f'Learning Curve for Fold no {bestModelIndex+1} | {selectMethod} Model with {selectTransform} Transformation', fontsize=20) 
    ax1.plot(list(zip(historyPerFold[bestModelIndex][0], historyPerFold[bestModelIndex][-2])), label=['Loss', 'Validation Loss'])
    ax1.set_xlabel('Epoch')
    ax2.plot(list(zip(historyPerFold[bestModelIndex][1], historyPerFold[bestModelIndex][-1])), label=['Accuracy', 'Validation Accuracy'])
    ax2.set_xlabel('Epoch')
    ax1.legend()
    ax2.legend()
    destinationFile = f'{savefolder}\{selectMethod} Model of Learning_curve_with_{selectTransform}.png'
    if os.path.isfile(destinationFile):
            os.remove(destinationFile)
            f.savefig(destinationFile)
    else:
            f.savefig(destinationFile)
    plt.close()

# %%
def plot_accuracy_per_fold(selectTransform, selectModel, historyPerFold, savefolder):
    plt.figure(figsize=(10, 6))
    for fold, history in enumerate(historyPerFold, 1):
        val_accuracy = history[3]  # Validation accuracy from history
        plt.plot(val_accuracy, label=f'Fold {fold}')

    plt.title(f'Accuracy for {selectModel} Model {selectTransform} Transformation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    destinationFile = f'{savefolder}\{selectModel} Model Accuracy with_{selectTransform}.png'
    if os.path.isfile(destinationFile):
            os.remove(destinationFile)
            plt.savefig(destinationFile)
    else:
            plt.savefig(destinationFile)
    plt.close()

# %%
def plot_auc_per_fold(selectModel, selectTransform, roc_per_fold, radioID, savefolder):
    plt.figure(figsize=(10, 6))
    for fold, roc_data in enumerate(roc_per_fold, 1):
        for i, (fpr, tpr) in enumerate(roc_data):
            label = ''
            if radioID[i] == 3:
                label = 'Co-60'
                plt.plot(fpr, tpr, label=f'Fold {fold}, {label}')
            elif radioID[i] == 4:
                label = 'Cs-137'
                plt.plot(fpr, tpr, label=f'Fold {fold}, {label}')
            elif radioID[i] == 5:
                label = 'Eu-152'
                plt.plot(fpr, tpr, label=f'Fold {fold}, {label}')


    plt.title('AOC Curve per Fold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    destinationFile = f'{savefolder}/{selectModel}_Model_AUC_with_{selectTransform}.png'
    if os.path.isfile(destinationFile):
        os.remove(destinationFile)
        plt.savefig(destinationFile)
    else:
        plt.savefig(destinationFile)
    plt.close()

# %%
def plot_classification_matrix_per_fold(selectModel, selectTransform, model_per_fold, inputs, outputs, savefolder):
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    inputs = np.array(inputs)[indices]
    outputs = np.array(outputs)[indices]
    for fold, model in enumerate(model_per_fold, 1):
        plt.figure(figsize=(8, 6))  # Adjust size if needed
        
        # Predict probabilities for each class
        predicted_probabilities = model.predict(inputs)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(outputs.argmax(axis=1), predicted_labels)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
                    xticklabels=['Co-60', 'Cs-137', 'Eu-152'], 
                    yticklabels=['Co-60', 'Cs-137', 'Eu-152']) 
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'{selectModel} with {selectTransform} Fold {fold} Classification Matrix')
        
        # Save figure with fold number in the file name
        destinationFile = f'{savefolder}/{selectModel}_Model_Classification_Matrix_with_{selectTransform}_Fold_{fold}.png'
        plt.savefig(destinationFile)
        plt.close()

# %%
# Commencing training for all colormapping and transformation combinations
for transform in transformMethod:
    for model in ModelMethod:
        totalData, totalLabel, radioID = data_preprocess(transform)
        preshuffled_data, preshuffled_label = preshuffle_data_label(totalData, totalLabel)
        inputs_dims = (np.shape(preshuffled_data)[1], np.shape(preshuffled_data)[2], np.shape(preshuffled_data)[3])
        rocPerFold, bestModelIndex, historyPerFold, modelPerFold = k_model_train(preshuffled_data, preshuffled_label, inputs_dims, transform, model, radioID)
        save_roc_plot(transform, model, bestModelIndex, radioID, radioTokenizer, atomTokenizer, rocPerFold, evaluatePath)
        save_learning_plot(transform, bestModelIndex, model, historyPerFold, evaluatePath)
        plot_accuracy_per_fold(transform, model, historyPerFold, evaluatePath)
        plot_auc_per_fold(model, transform, rocPerFold, radioID, evaluatePath)
        plot_classification_matrix_per_fold(model, transform, modelPerFold, preshuffled_data, preshuffled_label, evaluatePath)



