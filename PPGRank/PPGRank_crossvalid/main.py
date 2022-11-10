from Train import train
from Test import test
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
    ap.add_argument("-nf", "--features_no", required=True,
                    help="number of features in the dataset")
    ap.add_argument("-g", "--gamma", required=False, default=1,
                    help="Gamma value; default = 1")
    ap.add_argument("-alpha", "--learning_rate", required=False, default=0.005,
                    help="Learning rate of the agent; Defaults to 0.005")
    ap.add_argument("-dp", "--display", required=False, default=True,
                    help="Displays each query NDCG while calculating, else prints after entire query set calculated")
    args = vars(ap.parse_args())

    data_dir = str(args["data"])
    n_features = int(args["features_no"])
    epochs = int(args["epoch"])
    gamma = int(args["gamma"])
    alpha = float(args["learning_rate"])
    display = bool(args["display"])

    # theta = np.random.normal(scale=0.001, size=(25, 1))
    theta = np.array([0.1] * n_features)
    theta = np.reshape(theta, (n_features, 1))
    theta = np.round(normalize_minmax(theta) * 2 - 1, 4)

    print("Initial Param: ")
    print(list(theta))

    for fold in range(1, 6):
        print(f"\n--- For Fold {fold}---\n")
        # Current Fold
        training_set = data_dir + "/Fold" + str(fold) + "/train.txt"
        validation_set = data_dir + "/Fold" + str(fold) + "/vali.txt"
        testset = data_dir + "/Fold" + str(fold) + "/test.txt"

        # Find best lr using validation set
        lr = [0.01, 0.05, 0.09, 0.005]
        results_train_alpha = []
        results_valid_alpha = []
        theta_alpha = []
        results_alpha = []

        for alpha in lr:
            print("\n----Performing Training----\n")
            # Training and Validation
            theta_ret, iterative_results_train = train(training_set, theta, epochs, n_features, gamma, alpha, display)
            results_train_alpha.append(iterative_results_train)
            theta_alpha.append(theta_ret)
            print("\n----Performing Validation----\n")
            results_valid = test(theta_ret, validation_set, n_features, gamma, display)
            results_valid_alpha.append(results_valid)
            results_alpha.append(np.mean(results_valid["mean"][1:]))

        # Picking the best model
        best_alpha = lr[np.argmax(results_alpha)]
        print(f"\n--Obtained best alpha = {best_alpha} --\n")

        print("\n----Performing Testing----\n")
        # Using the optimized parameter on Testset
        results_test = test(theta_alpha[np.argmax(results_alpha)], testset, n_features, gamma, display)

        # Storing the values
        results = {'Training': [], 'Validation': [], 'Test': []}
        for iteration in results_train_alpha[np.argmax(results_alpha)]:
            results['Training'].append(results_train_alpha[np.argmax(results_alpha)][iteration]['mean'])
        for iteration in results_valid_alpha[np.argmax(results_alpha)]:
            results['Validation'].append(results_valid_alpha[np.argmax(results_alpha)]['mean'])
        results['Test'].append(results_test["mean"])

        # pickle current fold results for visualization later
        outfile = "fold_" + str(fold) + "_data"
        pickle_data(results, outfile)

        # Save weights
        outfile = "fold_" + str(fold) + "_weights"
        pickle_data(theta, outfile)
