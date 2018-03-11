import numpy as np  
import pandas as pd 


def get_mnist(limit=None):
	df = pd.read_csv("train.csv")
	data = df.as_matrix()

	np.random.shuffle(data)

	X = data[:, 1:] / 255
	Y = data[:, 0]

	if limit is not None:
		X, Y = X[:limit], Y[:limit]

	return X, Y