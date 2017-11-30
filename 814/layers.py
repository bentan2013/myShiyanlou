import numpy as np


class Data:
	
	def __init__(self, name, batch_size):
		
		with open(name, 'rb') as f:
			data = np.load(f)
		self.x = data[0]
		self.y = data[1]
		self.l = len(self.x)
		self.batch_size=batch_size
		self.pos = 0

	def forward(self):
		pos = self.pos
		bat = self.batch_size
		l = self.l
		if pos + bat >= 1:
			ret = (self.x[pos:l], self.y[[pos:l]
			self.pos = 0
			index = range(l)
			np.random.shuffle(index)
			self.x = self.x[index]
			self.y = self.y[index]
		else:
			ret = (self.x[pos:pos+bat], self.y[pos:pos+bat]
			self.pos += self.batch_size
		return ret, self.pos

	def backward(self, d):
		pass


	


