# Custom dataset to load CLIP embeddings from disk.
#  Files should contain embeddings as (1, E) or (E)

######## Folder Layout ########
#  Data                       #
#   |- CLIP                   #
#   |   |- test.npy <= eval   #
#   |   |- 01                 #
#   |   |   |- 000001.npy     #
#   |   |   |- 000002.npy     #
#   |   |   |   ...           #
#   |   |   |- 000999.npy     #
#   |   |   \- 001000.npy     #
#   |   |- 02                 #
#   |   |  ...                #
#   |   |- 09                 #
#   |   \- 10                 #
#   |- META <= other versions #
#     ...                     #
###############################

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

DEFAULT_ROOT = "data"
ALLOWED_EXTS = [".npy"]

class Shard:
	"""
	Shard to store embedding:score pairs in
		path: path to embedding on disk
		value: score for the original image
	"""
	def __init__(self, path, value):
		self.path = path
		self.value = value
		self.data = None

	def exists(self):
		return os.path.isfile(self.path) and self.value is not None

	def get_data(self):
		if self.data is not None: return self.data
		data = torch.from_numpy(
			np.load(self.path)
		)
		if data.shape[0] == 1:
			data = torch.squeeze(data, 0)
		assert not torch.isnan(torch.sum(data.float()))
		return {
			"emb": data,
			"val": torch.tensor([self.value]),
		}

	def preload(self):
		self.data = self.get_data()

class EmbeddingDataset(Dataset):
	def __init__(self, ver, root=DEFAULT_ROOT, preload=False):
		self.ver = ver
		self.root = root
		self.shards = []

		print("Dataset: Parsing data from disk")
		for cat in tqdm(os.listdir(f"{root}/{ver}")):
			cat_dir = f"{root}/{ver}/{cat}"
			if not os.path.isdir(cat_dir): continue
			for i in os.listdir(cat_dir):
				fname, ext = os.path.splitext(i)
				if ext not in ALLOWED_EXTS: continue
				self.shards.append(
					Shard(
						path = f"{root}/{ver}/{cat}/{fname}.npy",
						value = float(cat) / 10,
					)
				)

		if preload:  # cache to RAM
			print("Dataset: Preloading data to system RAM")
			[x.preload() for x in tqdm(self.shards)]

		print(f"Dataset: OK, {len(self)} items")

	def __len__(self):
		return len(self.shards)

	def __getitem__(self, index):
		return self.shards[index].get_data()

	def get_eval(self, size=None):
		shard = Shard(f"{self.root}/{self.ver}/test.npy", 1.0)
		data = shard.get_data() if shard.exists() else self[0]
		return {k:v.unsqueeze(0).to(torch.float32) for k,v in data.items()}
