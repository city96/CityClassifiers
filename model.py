import torch
import torch.nn as nn

class ResBlock(nn.Module):
	"""Linear block with residuals"""
	def __init__(self, ch):
		super().__init__()
		self.join = nn.ReLU()
		self.long = nn.Sequential(
			nn.Linear(ch, ch),
			nn.LeakyReLU(0.1),
			nn.Linear(ch, ch),
			nn.LeakyReLU(0.1),
			nn.Linear(ch, ch),
		)
	def forward(self, x):
		return self.join(self.long(x) + x)

class PredictorModel(nn.Module):
	"""Main predictor class"""
	def __init__(self, features=768, outputs=1, hidden=1024):
		super().__init__()
		self.features = features
		self.outputs = outputs
		self.hidden = hidden
		self.up = nn.Sequential(
			nn.Linear(self.features, self.hidden),
			ResBlock(ch=self.hidden),
		)
		self.down = nn.Sequential(
			nn.Linear(self.hidden, 128),
			nn.Linear(128, 64),
			nn.Dropout(0.1),
			nn.LeakyReLU(),
			nn.Linear(64, 32),
			nn.Linear(32, self.outputs),
		)
		self.out = nn.Softmax(dim=1) if self.outputs > 1 else nn.Tanh()
	def forward(self, x):
		y = self.up(x)
		z = self.down(y)
		if self.outputs > 1:
			return self.out(z)
		else:
			return (self.out(z)+1.0)/2.0
