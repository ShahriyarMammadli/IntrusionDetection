# Shahriyar Mammadli
# Import required modules
import helperFunctions as hf
import pickle
# Set dataset root path
rootPath = "C:/Users/smammadli/Desktop/Store/DataSets/Raw/User-Digital-Print/ISOT_Web_Interactions(Mouse_Keystroke_SiteAction)_Dataset"
# Set file format of the data
fileFormat = '.txt'
# Split the keystroke data into train and validation
# If train validation
def prepareData(trainValSplit = False):
    if trainValSplit:
        hf.trainValSplit(rootPath + "/Keystrokes/genuine", fileFormat, ratio=(0.7, 0.3), seed=2020)
        hf.trainValSplit(rootPath + "/MouseActions/genuine", fileFormat, ratio=(0.7, 0.3), seed=2020)
        hf.trainValSplit(rootPath + "/SiteActions/genuine", fileFormat, ratio=(0.7, 0.3), seed=2020)

    # Read the data set into a high-dimensional space
    labels, features = hf.readDataset(rootPath + "/Keystrokes/genuine/splitRes/train", fileFormat)
    # Write the processed data into a pickle file
    with open('trainData.pickle', 'wb') as handle:
        pickle.dump([labels, features], handle, protocol=pickle.HIGHEST_PROTOCOL)
