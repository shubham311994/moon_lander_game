"""
Contains all the code to read, preprocess the data.
"""
from neural_network import NeuralNetwork
from pathlib import Path
from csv import reader
import csv
import argparse
from sklearn.model_selection import train_test_split


def read_data(path: str) -> list:
    dataset = list()
    with open(Path(path), 'r') as file:
        csv_reader = reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def dataset_minmax(dataset):
    stats = [[max(column), min(column)] for column in zip(*dataset)]
    return stats


def normalize_dataset(dataset, minmax_value):
    for row in dataset:
        for i in range(len(row)):
            row[i] = -(row[i] - minmax_value[i][1]) / (minmax_value[i][1] - minmax_value[i][0])


def convert_to_features_and_target(dataset):
    features = []
    target = []
    for i in range(len(dataset)):
        features.append([dataset[i][0], dataset[i][1]])
        target.append([dataset[i][2], dataset[i][3]])
    return features, target


parser = argparse.ArgumentParser()
parser.add_argument("--path",
                    type=str,
                    help="Please specify the full path of the data file.",
                    required=True)
args = parser.parse_args()

if __name__ == "__main__":
    data = read_data(args.path)
    minmax = dataset_minmax(data)
    normalize_dataset(data, minmax)
    features_list, target_list = convert_to_features_and_target(dataset=data)
    # initializing the neural network with the required no of input, hidden & output nodes
    neural_network = NeuralNetwork(2, 12, 2)

    # split data into train and test/cross validation test set with 70% for training and 30% for
    # testing & validation
    X_train, X_Cross_Validation, Y_train, Y_Cross_Validation = train_test_split(features_list, target_list,
                                                                                test_size=0.3, shuffle=True)
    #  splitting the above test set data into half so each would amount of 15 % of total data"""
    X_test, X_val, Y_test, Y_val = train_test_split(X_Cross_Validation, Y_Cross_Validation, test_size=0.5,
                                                    shuffle=True)
    # train the neural network using the train data and validation data
    neural_network.train(X_train, Y_train, X_val, Y_val, 300)

    # testing on unseen dataset
    neural_network.test(X_test, Y_test)

    pass
