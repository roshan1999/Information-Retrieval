from multiprocessing_test import train
import numpy as np
import os
import pickle
from LoadData import normalize_minmax
import argparse


def pickle_data(data, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    output_handle = open(output_file, 'ab')
    pickle.dump(data, output_handle)
    output_handle.close()


if __name__ == '__main__':
    np.random.seed(7)

    # Constructing the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True,
                    help="relative or absolute directory of the directory where fold exists")
    ap.add_argument("-e", "--epoch", required=True,
                    help="number of epochs to run for")
    ap.add_argument("-f", "--fold", required=True,
                    help="which fold to run")
    ap.add_argument("-nf", "--features_no", required=True,
                    help="number of features in the dataset")
    ap.add_argument("-n", "--normalize", required=False, default=False,
                    help="True if features to be normalized; Defaults to False")
    ap.add_argument("-g", "--gamma", required=False, default=1,
                    help="Gamma value; default = 1")
    ap.add_argument("-alpha", "--learning_rate", required=False, default=0.005,
                    help="Learning rate of the agent; Defaults to 0.005")
    ap.add_argument("-np", "--num_process", required=False, default=4,
                    help="Number of processes to run in parallel")
    ap.add_argument("-dp", "--display", required=False, default=True,
                    help="Displays each query NDCG while calculating, else prints after entire query set calculated")
    args = vars(ap.parse_args())

    data_dir = str(args["data"])
    n_features = int(args["features_no"])
    fold = int(args["fold"])
    epochs = int(args["epoch"])
    gamma = int(args["gamma"])
    alpha = float(args["learning_rate"])
    display = bool(args["display"])
    num_process = int(args['num_process'])

    theta = np.random.normal(scale=0.001, size=(n_features, 1))
    # theta = np.array([0.1] * n_features)
    theta = np.reshape(theta, (n_features, 1))
    theta = np.round(normalize_minmax(theta) * 2 - 1, 4)

    # print("Initial Param: ")
    # print(list(theta))

    # Current Fold
    training_set = data_dir + "/Fold" + str(fold) + "/train.txt"
    validation_set = data_dir + "/Fold" + str(fold) + "/vali.txt"
    testset = data_dir + "/Fold" + str(fold) + "/test.txt"

    # Training and Validation
    theta, iterative_results_train, iterative_results_test, _ = train(validation_set,
                                                                    training_set, testset,
                                                                    theta, epochs, n_features,
                                                                    gamma, alpha, display, num_process)

    # print("Final Param")
    # print(list(theta))

    results = {'Training': [], 'Test': []}

    for iteration in iterative_results_train:
        results['Training'].append(iterative_results_train[iteration]['mean'])

    # for iteration in iterative_results_valid:
    #     results['Validation'].append(iterative_results_valid[iteration]['mean'])

    for iteration in iterative_results_test:
        results['Test'].append(iterative_results_test[iteration]['mean'])

    # results['Rewards'] = rewards_total

    # pickle current fold results for visualization later
    outfile = "fold_" + str(fold) + "_data"
    pickle_data(results, outfile)

    # Save weights
    outfile = "fold_" + str(fold) + "_weights"
    pickle_data(theta, outfile)
