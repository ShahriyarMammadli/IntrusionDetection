# Shahriyar Mammadli
# Import required modules
import os
from shutil import copy2
import numpy as np
import math
import csv

# Function to copy the given files to the given folder
def copyFilesToPath(source, destination, files):
    for sample in files:
        copy2(source + sample, destination)

# This function splits the data into train and validation sets
# Seed is 1 in default
# First element inside of ratio is accepted as train set ratio and...
# ...second one is considered as validation set ratio
def trainValSplit(path, fileformat, ratio, seed=1):
    if(sum(list(ratio)) != 1):
        raise ValueError("Sum of the ratio elements should be equal to 1")
    # Set a random seed
    np.random.seed(seed)
    # Create a folder where the split files will be written
    if not os.path.exists(path + '/splitRes'):
        os.makedirs(path + '/splitRes')
    if not os.path.exists(path + '/splitRes/train'):
        os.makedirs(path + '/splitRes/train')
    if not os.path.exists(path + '/splitRes/validation'):
        os.makedirs(path + '/splitRes/validation')
    # Iterate over the users in a given folder
    for folder in os.listdir(path):
        # Make sure the folder name starts with 'user'
        if folder.startswith('user'):
            # Create a user folder inside of the output folders
            if not os.path.exists(path + '/splitRes/train/' + folder):
                os.makedirs(path + '/splitRes/train/' + folder)
            if not os.path.exists(path + '/splitRes/validation/' + folder):
                os.makedirs(path + '/splitRes/validation/' + folder)
            # Make sure that right data pieces are considered by checking their format
            samples = [elem for elem in os.listdir(path + '/' + folder) if elem.endswith(fileformat)]
            # We give priority to training samples, thus we use ceil() function
            sizeOfTrain = int(math.ceil(len(samples) * list(ratio)[0]))
            trainSamples = np.random.choice(samples, sizeOfTrain, replace=False)
            # Remove those selected elements from the list which will give us validation set
            validationSamples = [i for i in samples if i not in trainSamples]
            # Add train samples to the relevant path
            copyFilesToPath(path + '/' + folder + '/', path + '/' + 'splitRes/' + 'train/' + folder, trainSamples)
            # Add validation samples to the relevant path
            copyFilesToPath(path + '/' + folder + '/', path + '/' + 'splitRes/' + 'validation/' + folder, validationSamples)

# Read the values from the lines of a given file
def readFromFile(filepath, delimiter = '\t'):
    # Initialize a feature vector and label
    array2D = []
    label = -1
    with open(filepath) as fileObject:
        # Skip the very first line which denotes the name of variables
        lines = csv.reader(fileObject, delimiter=delimiter)
        headers = next(lines)
        lines = list(lines)
        # If the second line is empty throw and exception
        try:
           label = lines[0][0]
        except:
            print("Corrupted file detected, it will be deleted")
        for line in lines:
            array2D.append(line[1:3])
    return label, array2D

# Remove the file in a given path
def removeFile(path):
    os.remove(path)

# This function reads files from the given path and deletes a...
# ...file which is empty or has few data points
def readDataset(path, fileformat):
    # Initialize labels and features vector
    labels = []
    features = []
    # Iterate over folders
    for folder in os.listdir(path):
        # Make sure the folder name starts with 'user'
        if folder.startswith('user'):
            # Make sure that right data pieces are considered by checking their format
            samples = [elem for elem in os.listdir(path + '/' + folder) if elem.endswith(fileformat)]
            for sample in samples:
                label, array2D = readFromFile(path + '/' + folder + '/' + sample)
                # Check if the file is in right format or not. i.e. if it returns -1...
                # ...then it is somehow corrupted, thus delete that file
                if(label == -1):
                    removeFile(path + '/' + folder + '/' + sample)
                else:
                    labels.append(label)
                    features.append(array2D)
    return labels, features

