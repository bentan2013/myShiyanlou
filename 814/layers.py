import numpy as np


class Data:
	
	def __init__(self, name, batch_size):
		
		with open(name, 'rb') as f:
			data = np.load(f)
		self.x = data[0]
		self.y = data[1]

