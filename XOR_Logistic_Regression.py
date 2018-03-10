'''
Created_by : Anand Tiwari
Created_at : 15/08/2018

Description: This algorithm describe the working or implimentation of logistic regression.
			 We have used XOR problem to impliment this algorithm. We have used sigmoid (also known as Logistic)
			 as a activation function and Gradient Descent for training the model.
'''

import numpy as np 
import matplotlib.pyplot as plt 


# Sigmoid function 
def sigmoid(z):
	return 1 / (1 + np.exp(-z))


# Cost or Error function
def cross_entropy(T, y):
	return -(T * np.log(y) + (1 - T) * np.log(1 - y)).mean() 


# Classification of accurate prediction
def classification(y, prediction):
	return np.mean(y == prediction)


# main function contain main logic
def main():
	N = 4
	D = 2

	# The XOR inputs
	X = np.array([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]
		])

	# Target
	T = np.array([0, 1, 1, 0])

	# Feature Engineering 
	ones = np.array([[1] * N]).T
	xy = np.matrix(X[:, 0] * X[:, 1]).T
	Xb = np.array(np.concatenate((ones, xy, X), axis = 1))

	# randomly initialize weights
	w = np.random.randn(D + 2)

	y = sigmoid(Xb.dot(w))

	learning_rate = 0.1

	error = []

	# Gradient Descent
	for i in range(5000):
		e = cross_entropy(T, y)
		error.append(e)

		if i % 100 == 0:
			print("Error in {0} iteration is {1}".format(i, e))

		w = w - learning_rate * (Xb.T.dot(y - T) + 0.01 * w)
		y = sigmoid(Xb.dot(w))

	print("\nFinal weights are", w)
	print("\nFinal classification result", classification(T, np.round(y)))

	plt.plot(error)
	plt.show()

if __name__ == '__main__':
	main()