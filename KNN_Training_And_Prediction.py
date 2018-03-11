'''
Created_by : Anand Tiwari
Created_At : 20/02/2018

Description : This is the implimnetation of  K-NeirestNeighbour Algorithm. I have used MNIST data to 
			  to impliment this algorithm. KNN is both classification and regression algorithm. 
			  In classification, we look on the neirest point and then find which class has more 
			  occurance in those neirest points. In regression, take mean of the neirest points. 

			  In this algorithm fit only stores the input and do nothing all work is done by the
			  predict. So, it is also called as Lazy Learner. Distance matrix used is Euclidean.
			  I have used one list(sl1) to store tuple of [(distance, class)]. list contain number of item equal to K.
			  From that list i had made a dictionary(votes) of {class : number of occurance in the list}.
			  With this dictionary classification is or prediction is done.
'''


import numpy as np 
from utils import get_mnist 
from sortedcontainers import SortedList
from datetime import datetime 
import matplotlib.pyplot as plt


class KNN(object):

	# Initializing nunber of neirest point to be used
	def __init__(self, k):
		self.k = k


	# fit function only stores the inputs data
	def fit(self, X, y):
		self.X = X
		self.y = y


	# predict function perform all the work of finding neirest points and classification
	def predict(self, X):

		# defining output matrix
		y = np.zeros(len(X))

		# loop for every single point in test data
		for i, x in enumerate(X):

			# creating a list to store neirest point as a tuple (distance, class)
			sl = SortedList(load = self.k)

			# loop : single test data point against every input data point 
			for j, xt in enumerate(self.X):

				# measuring distance 
				diff = x - xt
				d = diff.dot(diff)

				# adding item to list 
				# on every iteration we check length of list
				# It must be equal to K
				if len(sl) < self.k:
					sl.add((d, self.y[j]))
				else:
					if d < sl[-1][0]:
						del sl[-1]
						sl.add((d, self.y[j]))
			
			# dictionary to store {class : occurance}
			votes = {}
			
			for _, v in sl: # v is class 
				# print(v)
				votes[v] = votes.get(v, 0) + 1
				# print(votes.get(v, 0)

			# dummies
			max_count = 0
			max_value_count = -1

			# looping through every item in the dictionary
			# to get most occured class
			for v, count in votes.items():
				if count > max_count:
					max_count = count
					max_value_count = v

			# assigning the class to that index
			y[i] = max_value_count

		return y


	# Score function gives accuracy of prediction
	def score(self, X, y):
		pred = self.predict(X)
		return np.mean( y == pred)


if __name__ == '__main__':
	
	# getting the data (NOTE: I have used only 2000 data point.)
	# You can change this 
	X, y = get_mnist(2000)

	Ntrain = int(0.7 * len(X))
	# Train and Test splits 
	Xtrain, ytrain = X[:Ntrain], y[:Ntrain]
	Xtest, ytest = X[Ntrain:], y[Ntrain:]
	train_score = []
	test_score = []


	# choosing optimal K is difficult
	# I have used K = 1 to 5
	# You can change K as per your requirement 
	for k in (1, 2, 3, 4, 5):

		t0 = datetime.now()
		knn = KNN(k)
		knn.fit(Xtrain, ytrain)
		print("Training Time: ", datetime.now() - t0)

		t0 = datetime.now()
		print("Training Accuracy: ", knn.score(Xtrain, ytrain))
		print("Time to compute train accuracy :", datetime.now() - t0)
		train_score.append(knn.score(Xtrain, ytrain))

		t0 = datetime.now()
		print("Testing Accuracy: ", knn.score(Xtest, ytest))
		print("Time to compute test accuracy :", datetime.now() - t0)
		test_score.append(knn.score(Xtest, ytest))

		print("\n")


	# plottings
	plt.plot(train_score, label = 'Train Score')
	plt.plot(test_score, label = 'Test Score')
	plt.xlim(1, 6)
	plt.xticks( range(1, 6) )
	plt.legend()
	plt.show()
