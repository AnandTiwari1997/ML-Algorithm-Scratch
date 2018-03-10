'''
Created_by : Anand Tiwari
Created_at : 09/03/2018

Description: This algorithm describe the working or implimentation of an Artificial Neural
			 Network with input layer, 2 hidden layer and output layer. Sigmoid and Softmax
			 are used as a activation function. In this implimentation we used concept of Gradient
			 Ascent instead of Gradient Descent.
'''

import numpy as np 
import matplotlib.pyplot as plt 


# Defining the Softmax activation function : This function is used for output layer.
def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis = 1, keepdims = True)


# Sigmoid activation function : This function is used for the hidden layer.
def sigmoid(a):
	return 1 / (1 + np.exp(-a))


# Forward function for finding the probabilities or output
def forward(X, W1, b1, W2, b2, W3, b3):
	Z1 = sigmoid(X.dot(W1) + b1)
	Z2 = sigmoid(Z1.dot(W2) + b2)
	return softmax(Z2.dot(W3) + b3), Z1, Z2


# Derivative of Weight2 for the 1st hidden layer
# updating the weight2 
def derivative_W2(Z1, Z2, T, Y, W3):
	# N, D = Z1.shape
	# M, K = W2.shape

	# slow
	# ret1 = np.zeros((D, M))

	# for n in range(N):
	# 	for k in range(K):
	# 		for m in range(M):
	# 			for d in range(D):
	# 				ret1[d, m] += (T[n, k] - Y[n, k]) * W2[m, k] * Z[n, m] * (1 - Z[n, m]) * X[n, d]

	# Vectorized operations are faster
	dz = (T - Y).dot(W3.T) * Z2 * (1 - Z2)
	return Z1.T.dot(dz)

# derivation of bias2 for the first hidden layer
# updating bias2
def derivative_b2(T, Y, W3, Z2):
	return ((T - Y).dot(W3.T) * Z2 * (1 - Z2)).sum(axis = 0)

# derivative of weight3 for the second hiden layer
# updating the weight3
def derivative_W3(Z2, T, Y):
	# N, K = T.shape
	# M = Z.shape[1]

	# slow
	# ret1 = np.zeros((M, K))

	# for n in range(N):
	# 	for m in range(M):
	# 		for k in range(K):
	# 			ret1[m, k] += (T[n, k] - Y[n, k]) * Z[n, m] 

	# little faster
	# ret2 = np.zeros((M, K))

	# for n in range(N):
	# 	for k in range(K):
	# 		ret2[:, k] += (T[n, k] - Y[n, k]) * Z[n, :]

	# Faster 
	# ret3 = np.zeros((M, K))

	# for n in range(N):
	# 	ret3 += np.outer(Z[n], T[n] - Y[n])

	# Vetorized operations are faster
	# ret4 = Z.T.dot(T - Y)

	return Z2.T.dot(T - Y)


# derivation of bias3 for the second hidden layer
# updating bias3
def derivative_b3(T, Y):
	return (T - Y).sum(axis = 0)


# derivation of weight1 for the input layer
# updating weight1
def derivative_W1(X, Z1, Z2, T, Y, W3, W2):
	dz = ((T - Y).dot(W3.T) * Z2 * (1 - Z2)).dot(W2.T) * Z1 * (1 - Z1)
	return X.T.dot(dz)


# derivation of bias1 for the input layer
# updating bias1
def derivative_b1(T, Y, W3, W2, Z2, Z1):
	return (((T - Y).dot(W3.T) * Z2 * (1 - Z2)).dot(W2.T) * Z1 * (1 - Z1)).sum(axis = 0)

 
# Cost or Error function
def cross_entropy(T, Y):
	return -np.mean(T * np.log(Y))


# Classification of accurate predictions
def classification_rate(y, y_hat):
	correct = 0
	total = 0
	for i in range(len(y)):
		total += 1
		if y[i] == y_hat[i]:
			correct += 1
	return float(correct) / total


# main function contains main logic
def main():
	N = 500
	D = 2
	M = 3
	K = 3

	# Three Guassian Cloud of data
	X1 = np.random.randn(N, D) + np.array([0, -2]) 
	X2 = np.random.randn(N, D) + np.array([2, 2]) 
	X3 = np.random.randn(N, D) + np.array([-2, 2]) 
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*N + [1]*N + [2]*N)
	
	N_Y = len(Y)
	T = np.zeros((N_Y, K))

	for i in range(N_Y):
		T[i, Y[i]] = 1

	plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
	plt.show()

	# randomly initialize weights and bias
	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, M)
	b2 = np.random.randn(M)
	W3 = np.random.randn(M, K)
	b3 = np.random.randn(K)

	learning_rate = 0.0001
	cost = []

	# Gradient Ascent
	for i in range(100000):
		output, hidden1, hidden2 = forward(X, W1, b1, W2, b2, W3, b3)
		if i%100 == 0:
			c = cross_entropy(T, output)
			p = np.argmax(output, axis = 1)
			accuracy = classification_rate(Y, p)
			print("Iteration :", i, "Cost :", c, "Accuracy :", accuracy)
			cost.append(c)

		W3 += learning_rate * derivative_W3(hidden2, T, output) # Second hidden Layer
		b3 += learning_rate * derivative_b3(T, output)
		W2 += learning_rate * derivative_W2(hidden1, hidden2, T, output, W3) # First hidden layer
		b2 += learning_rate * derivative_b2(T, output, W3, hidden2)
		W1 += learning_rate * derivative_W1(X, hidden1, hidden2, T, output, W3, W2) # Inout layer
		b1 += learning_rate * derivative_b1(T, output, W3, W2, hidden2, hidden1)

	plt.plot(cost)
	plt.show()

if __name__ == "__main__":
	main()