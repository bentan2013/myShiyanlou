import numpy as np
from layers import *


class FullyConnect:

	def __init__(self, l_x, l_y):
		self.weights = np.random.randn(l_y, l_x)
		self.bias = np.random.randn(l_y, 1)
		self.lr = 0

	def forward(self, x):
		self.x = x
		self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])
		return self.y

	def backward(self, d):
		self.ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]
		self.dw = np.sum(self.ddw, axis=0) / self.x.shape[0]
 		self.db = np.sum(d, axis=0) / self.x.shape[0]
		self.dx = np.array([np.dot(self.weights.T, dd) for dd in d]) 	
		self.weights -= self.lr * self.dw
		self.bias -= self.lr * self.db
		return self.dx


class Sigmoid:
	def __init__(self):
		pass

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def forward(self, x):
		self.x = x
		self.y = self.sigmoid(x)
		return self.y
	
	def backward(self, d):
		sig = self.sigmoid(self.x)
		self.dx = d * sig * (1 - sig)
		return self.dx

	
class QuadraticLoss:

	def __init__(self):
		pass

	def forward(self, x, label):
		self.x = x
		self.label = np.zeros_like(x)
		for a, b in zip(self.label, label):
			a[b] = 1.0
		self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0]/2
		return self.loss

	def backward(self):
		self.dx = (self.x - self.label) / self.x.shape[0]
		return self.dx


class Accuracy:
	
	def __init__(self):
		pass

	def forward(self, x, label):
		self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])
		self.accuracy = 1.0 * self.accuracy / x.shape[0]
		return self.accuracy


def main():
	datalayer1 = Data('../../train.npy', 1024)
	datalayer2 = Data('../../validate.npy', 10000)
	
	inner_layers = []
	inner_layers.append(FullyConnect(17*17, 26))
	inner_layers.append(Sigmoid())
	losslayer = QuadraticLoss()
	accuracy = Accuracy()

	for layer in inner_layers:
		layer.lr = 1000.0
	
	epoches = 20
	for i in range(epoches):
		print 'epoches:' , i
		losssum = 0
		iters = 0
		
		while True:
			data, pos = datalayer1.forward()
			x, label = data
			for layer in inner_layers:
				x = layer.forward(x)
			
			loss = losslayer.forward(x, label)
			losssum += loss
			iters += 1
			d = losslayer.backward()
			for layer in inner_layers[::-1]:
				d = layer.backward(d)
			if pos == 0:
				data, _ = datalayer2.forward()
				x, label = data
				for layer in inner_layers:
					x = layer.forward(x)
				accu = accuracy.forward(x, label)
				print 'loss', losssum / iters
				print 'accuracy:', accu
				break
	

if __name__=='__main__':
	main()
	
	
			
		


        

	
