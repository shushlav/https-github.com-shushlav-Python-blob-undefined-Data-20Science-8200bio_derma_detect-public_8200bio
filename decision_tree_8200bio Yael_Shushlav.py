# 8200 Bio Data Challenge 2021 - DermaDetect Ltd.
# Dear participant,
# This file is provided for guidance and framework, for your convenience.
# Feel free to modify it at will, add (well-documented) input arguments and functionalities, etc.
# ... so long the file is able to be evaluated as described in the instructions.
# Official submission is defined as the last commit to your personal branch
# at the challenge's end time.

# Enjoy your journey to the world of teledermatology & AI diagnostics, and good luck to all!
# DermaDetect

# DermaDetect Copyright (C), 2021

from sklearn import tree
import pandas as pd
import pickle
import argparse
import os
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
# file names
dir_name = 'F:\Python\Data Science\8200bio_derma_detect\public_8200bio_challenge-main\data'
data_filename = 'dd_data.csv'
info_filename = 'data_info.csv'

# load data
data = pd.read_csv(os.path.join(dir_name, data_filename))
data_info = pd.read_csv(os.path.join(dir_name, info_filename), index_col=0)

class DecisionTreeTrainer:
    """ A class for training a decision tree on DermaDetect data """
    trained_model_filename = 'trained_model.pkl'
    data_relative_path = 'data/dd_data.csv'

    def __init__(self):
        self.main_dir = os.path.dirname(os.path.abspath(__file__))

        # TODO Fine-tune the decision tree classifier's parameters here
        self.model = tree.DecisionTreeClassifier()

    def load_data(self, input_file):
        data = pd.read_csv(input_file)
        preprocessed_data = self.preprocess_data(data)
        return preprocessed_data

    def load_training_data(self):
        input_file = os.path.join(self.main_dir, self.data_relative_path)
        return self.load_data(input_file)

    def save_model(self):
        filename = os.path.join(self.main_dir, self.trained_model_filename)
        print(filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        filename = os.path.join(self.main_dir, self.trained_model_filename)
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        self.model = model

    def preprocess_data(self, data):
                    # TODO - done
        ## drop features with missing data (over 50% missing data)
        data = data.dropna( axis=1, thresh=450)
        # devide columns according to dtype
        obj_columns = data.select_dtypes(include=['object']).columns
        num_columns = data.select_dtypes(include=['float64']).columns
        bool_columns = data.select_dtypes(include=['bool']).columns
        # Convert booleans into integers
        data[(bool_columns)] = data[(bool_columns)] * 1
        # Label Encoder for Gender
        ohe = LabelEncoder()
        ohe.fit_transform(data[['gender']])
        
        #  Creating an ordered list of unique values for the feature
        sizeCat = ['unknown', 'pea', 'handwatch', 'palm', '>palm']
        shapeCat = ['unknown', 'round', 'non-round']
        quantityCat = ['unknown', 'single', 'clustered', 'multiple', 'widespread']
        topographyCat = ['unknown', 'below', 'skin_level', 'above']
        hairLossCat = ['focal', 'diffused']
        #initiating the process  
        ordi = OrdinalEncoder(categories=[sizeCat, shapeCat, quantityCat, topographyCat, hairLossCat])    #initiating the process    
        ordi.fit_transform(data[['size', 'shape', 'quantity', 'topography', 'lossOfHair.type']])

        # Replacing the data with the encoder
        data[['size', 'shape', 'quantity', 'topography', 'lossOfHair.type']] = ordi.fit_transform(data[['size', 'shape', 'quantity', 'topography', 'lossOfHair.type']])
    
        #LabelEncode Target
        le = LabelEncoder()     # initialize
        data['diagnosis'] = le.fit_transform(data['diagnosis'])
        
        return data

    def train(self, inputs, outputs):
        self.model.fit(inputs, outputs)
        self.save_model()

    def evaluate(self, inputs, outputs):
        # TODO: call self.model.predict()
        # TODO: accurately compute the amount of correct predictions over the inputs size in %
        # compute classification accuracy for the logistic regression model
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
        pass


################################################################################
def get_cmd_args():
    """ Input arguments """
    args = argparse.ArgumentParser('DermaDetect - 8200 Data challenge 2021 - Decision tree diagnostics')
    args.add_argument('--evaluate_csv', '-e', type=str, default='', help='Path to the test dataset')
    return args.parse_args()


################################################################################
def main():
    args = get_cmd_args()
    do_train = len(args.evaluate_csv) == 0

    the_tree = DecisionTreeTrainer()

    if do_train:
        the_tree.load_training_data()
        # TODO split data into train & test groups
        the_tree.save_model()

        # TODO evaluate on test group

    else:  # evaluate
        the_tree.load_model()
        test_data = the_tree.load_data(args.evaluate_csv)

        # TODO evaluate
        # TODO optionally save, display, etc.

        # Lastly, print test accuracy
        test_accuracy_percent = 0.0  # EDIT ME (undoubtedly you'll do better than this :)
        print("Test accuracy: {:.2f}%".format(test_accuracy_percent))  # in [0-100], eg 99.99%


################################################################################
if __name__ == '__main__':
    main()
