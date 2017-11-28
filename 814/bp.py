import numpy as np

class FullyConnect:
	def __init__(self, l_x, l_y):
		self.weights = np.random.randn(l_x, l_y)
		self.bias = np.random.randn(1)

	def forward(self, x):
		self.x = x
		self.y = np.dot(self.weights, x) + self.bias
		return self.y

	def backward(self, d):
		self.dw = d * self.x
		self.db = d
		self.dx = d * self.weights
		return self.dw, self.db
		


        

	
