# Custom dataset to load CLIP embeddings from disk.
#  Files should contain embeddings as (1, E) or (E)

######## Folder Layout ########
#  Data                       #
#   |- CLIP                   #
#   |   |- test.npy <= eval   #
#   |   |- 0_test.npy <= cls  #
#   |   |- 01                 #
#   |   |   |- 000001.npy     #
#   |   |   |- 000002.npy     #
#   |   |   |   ...           #
#   |   |   |- 000999.npy     #
#   |   |   \- 001000.npy     #
#   |   |- 02_optional_name   #
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
from copy import deepcopy
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
		if self.data is not None: return deepcopy(self.data)
		data = torch.from_numpy(
			np.load(self.path)
		)
		if data.shape[0] == 1:
			data = torch.squeeze(data, 0)
		assert not torch.isnan(torch.sum(data.float()))
		return {
			"emb": data,
			"raw": self.value,
			"val": torch.tensor([self.value]),
		}

	def preload(self):
		self.data = self.get_data()

class EmbeddingDataset(Dataset):
	def __init__(self, ver, root=DEFAULT_ROOT, mode="class", preload=False):
		"""
		Main dataset that returns list of requested images as (C, E) embeddings
		  ver: CLIP version (folder)
		  root: Path to folder with sorted files
		  mode: Model type. Class pads return val to length of labels.
		  preload: Load all files into memory on initialization
		"""
		self.ver = ver
		self.root = f"{root}/{ver}"
		self.mode = mode
		self.shard_class = Shard

		if self.mode == "score":
			self.parse_shards(
				vprep = lambda x: float(x),
				norm  = True,
			)
			self.eval_data = self.get_score_eval()
		elif self.mode == "class":
			self.parse_shards(
				vprep = lambda x: int(x)
			)
			self.parse_labels()
			self.eval_data = self.get_class_eval()
		else:
			raise NotImplementedError("Unknown mode")

		if preload:  # cache to RAM
			print("Dataset: Preloading data to system RAM")
			[x.preload() for x in tqdm(self.shards)]

		print(f"Dataset: OK, {len(self)} items")

	def __len__(self):
		return len(self.shards)

	def __getitem__(self, index):
		data = self.load_shard(self.shards[index])
		data["index"] = index
		return data

	def load_shard(self, shard):
		return shard.get_data()

	def parse_shards(self, vprep, exts=ALLOWED_EXTS, norm=False):
		print("Dataset: Parsing data from disk")
		self.shards = []
		for cat in tqdm(os.listdir(self.root)):
			cat_dir = f"{self.root}/{cat}"
			if not os.path.isdir(cat_dir): continue
			for i in os.listdir(cat_dir):
				fname, ext = os.path.splitext(i)
				if ext not in exts: continue
				self.shards.append(
					self.shard_class(
						path = f"{self.root}/{cat}/{fname}{ext}",
						value = vprep(cat.split('_', 1)[0]),
					)
				)
		if norm:
			shard_min = min([x.value for x in self.shards])
			shard_max = max([x.value for x in self.shards])
			print(f"Normalizing scores [{shard_min}, {shard_max}]")
			for s in self.shards:
				s.value = (s.value - shard_min) / (shard_max - shard_min)

	def parse_labels(self):
		assert self.mode == "class"
		labels = list(set([int(x.value) for x in self.shards]))
		self.num_labels = len(labels)
		assert all([x in labels for x in range(self.num_labels)]), "Dataset: Class labels not sequential!"
		print(f"Dataset: Found {self.num_labels} separate classes")

	def get_class_eval(self, ext="npy"):
		out = [self.get_single_class_eval(x, ext) for x in range(self.num_labels)]
		return {
			"emb": torch.stack([x.get("emb") for x in out], dim=0),
			"val": torch.stack([x.get("val") for x in out], dim=0),
		}

	def get_single_class_eval(self, label, ext="npy"):
		fname = f"{label}_test.{ext}" if label >= 0 else f"test.{ext}"
		shard = self.shard_class(f"{self.root}/{fname}", label)
		if shard.exists():
			data = self.load_shard(shard)
		else:
			print(f"Dataset: Eval '{fname}' missing!")
			data = self[[x for x in range(len(self)) if self.shards[x].value == label][0]]
		val = torch.zeros(self.num_labels)
		val[label] = 1.0
		return {
			"emb": data.get("emb"),
			"val": val,
		}

	def get_score_eval(self, ext="npy"):
		shard = self.shard_class(f"{self.root}/test.{ext}", 1.0)
		data = self.load_shard(shard) if shard.exists() else self[0]
		return {
			"emb": data.get("emb").unsqueeze(0).to(torch.float32),
			"val": data.get("val").unsqueeze(0).to(torch.float32),
		}

################################
#    Code for live encoding    #
################################
import torchvision.transforms as TF
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

IMAGE_ROOT = "ratings"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif"]

class ImageShard(Shard):
	"""
	Shard to store embedding:score pairs in
		path: path to embedding on disk
		value: score for the original image
	"""
	def get_data(self):
		if self.data is not None: return deepcopy(self.data)
		return {
			"img": Image.open(self.path).convert("RGB"),
			"raw": self.value,
			"val": torch.tensor([self.value]),
		}

class ImageDataset(EmbeddingDataset):
	def __init__(self, ver, root=IMAGE_ROOT, mode="class", clip_dtype=torch.float16, preload=False):
		"""
		Secondary dataset that returns list of requested images as (C, E) embeddings
		  ver: CLIP version 
		  root: Path to folder with sorted files
		  mode: Model type. Class pads return val to length of labels.
		"""
		self.ver = ver
		self.root = root
		self.mode = mode

		self.device = "cuda"
		self.clip_ver = "openai/clip-vit-large-patch14-336"
		self.clip_dtype = clip_dtype
		self.shard_class = ImageShard

		assert self.ver in ["CLIP"], "Dataset: META Clip not supported for live encoding!"
		self.proc, self.clip = self.init_clip()

		if self.mode == "score":
			self.tfs = -1
			self.tf = None
			self.parse_shards(
				vprep  = lambda x: float(x),
				exts   = IMAGE_EXTS,
				norm   = True,
			)
			self.eval_data = self.get_score_eval(ext="png")
		elif self.mode == "class":
			self.tfs = self.proc.size.get("shortest_edge", 256)*2
			self.tf = TF.RandomCrop(self.tfs)
			self.parse_shards(
				vprep  = lambda x: int(x),
				exts   = IMAGE_EXTS,
			)
			self.parse_labels()
			self.eval_data = self.get_class_eval(ext="png")
		else:
			raise NotImplementedError("Unknown mode")

		[x.preload() for x in tqdm(self.shards)]
		print(f"Dataset: OK, {len(self)} items")

	def load_shard(self, shard):
		data = shard.get_data()
		img = data.pop("img")
		if self.tf and min(img.size) >= self.tfs:
			img = self.tf(img) # apply transforms
		data["emb"] = self.get_clip_emb(img).squeeze(0)
		return data

	def init_clip(self):
		print(f"Dataset: Initializing CLIP ({self.ver})")
		proc = CLIPImageProcessor.from_pretrained(self.clip_ver)
		clip = CLIPVisionModelWithProjection.from_pretrained(
			self.clip_ver,
			device_map  = self.device,
			torch_dtype = self.clip_dtype,
		)
		return (proc, clip)

	def get_clip_emb(self, raw):
		img = self.proc(
			images = raw,
			# do_rescale = False,
			return_tensors = "pt"
		)["pixel_values"].to(self.clip_dtype).to(self.device)
		with torch.no_grad():
			emb = self.clip(pixel_values=img)
		return emb["image_embeds"].detach().to(torch.float32)
