'''
Created_by : Anand Tiwari
Created_at : 15/08/2018

Description: This algorithm describe the working or implimentation of logistic regression.
			 We have used XOR problem to impliment this algorithm. We have used sigmoid (also known as Logistic)
			 as a activation function and Gradient Descent for training the model.
'''

import numpy as np 
import matplotlib.pyplot as plt 


class LogisticRegression(object):

	# initializing the hyperparameter while creatng object
	def __init__(self, reg = 0.01, learning_rate = 0.1, max_iters = 5000):
		self.reg = reg
		self.learning_rate = learning_rate
		self.max_iters = max_iters

	# Sigmoid function 
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))


	# Cost or Error function
	def cross_entropy(self, T, y):
		return -(T * np.log(y) + (1 - T) * np.log(1 - y)).mean() 


	# Classification of accurate prediction
	def classification(self, y, prediction):
		return np.mean(y == prediction)


	# fit function 
	def fit(self, X, Y):

		N, D = X.shape
		
		# randomly initialize weights
		self.w = np.random.randn(D)

		self.error = []

		# Gradient Descent
		for i in range(self.max_iters):
			YPred = self.sigmoid(X.dot(self.w))
			e = self.cross_entropy(Y, YPred)
			self.error.append(e)
			self.w -= self.learning_rate * (X.T.dot(YPred - Y) + self.reg * self.w)
			

	# predict function
	def predict(self, X):
		return self.sigmoid(X.dot(self.w))


	# score function
	def score(self, X, Y):
		YPred = self.predict(X)
		return self.classification(Y, np.round(YPred))


	# weight function to return the optimal weights
	def weights(self):
		return self.w


	# plot function to plot the error
	def plot(self):
		plt.plot(self.error, c='b', label = 'Train Error')
		plt.legend()
		plt.show()

if __name__ == '__main__':
	
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
	Y = np.array([0, 1, 1, 0])

	# Feature Engineering 
	ones = np.array([[1] * N]).T
	xy = np.matrix(X[:, 0] * X[:, 1]).T
	Xb = np.array(np.concatenate((ones, xy, X), axis = 1))

	# Creating model
	# I have used XOR. You can use any problem you want
	# Be careful with the input you provide
	lr = LogisticRegression()
	lr.fit(Xb, Y)
	print("classification Accuracy is : ", lr.score(Xb, Y))
	print("Optimal Weights are : ", lr.weights())
	lr.plot()