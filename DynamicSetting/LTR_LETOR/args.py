import argparse

def fetchArgs():
	ap = argparse.ArgumentParser()
	ap.add_argument("-fname", "--filename", required=True, help="Dataset to load")
	ap.add_argument("-M", "--init_docs", required=True, help="Number of documents to begin with")
	ap.add_argument("-extra", "--extra_docs", required=True, help="Number of documents to add at each timestep")
	ap.add_argument("-d", "--decay", required=True, help="Value to decay with")
	ap.add_argument("-nf", "--features_no", required=True, help="number of features in the model")
	ap.add_argument("-g", "--gamma", required=False, default=1, help="Gamma value; default = 1")
	ap.add_argument("-lr", "--learning_rate", required=False, default=0.001, help="Maximum Episode Length")
	ap.add_argument("-alpha", "--alpha", required=False, default=0.1, help="hyperparameter for SS in retrieval")
	ap.add_argument("-beta", "--beta", required=False, default=0.01, help="hyperparameter for RS in retrieval")
	ap.add_argument("-window", "--window_size", required=False, default=20000, help="Size of the window")
	args = vars(ap.parse_args())

	N = 40
	fname = str(args['filename'])
	M = int(args['init_docs'])
	decay_value = float(args['decay'])
	len_add_docs = int(args['extra_docs'])
	gamma = float(args['gamma'])
	alpha = float(args['alpha'])
	beta = float(args['beta'])
	num_features = int(args['features_no'])
	learning_rate = float(args['learning_rate'])
	window_size = int(args['window_size'])

	return M, decay_value, len_add_docs, N, gamma, num_features, learning_rate, window_size, fname