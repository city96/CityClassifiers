import os
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from model import AestheticPredictorModel

class CityAestheticsPipeline:
	"""
	Demo model pipeline for [image=>score] prediction
		Accepts a single model path on initialization.
		Resulting object can be called directly with a PIL image as the input
		Returns a single float value with the predicted score [0.0;1.0].
	"""
	clip_ver = "openai/clip-vit-large-patch14"
	def __init__(self, model_path, device="cpu", clip_dtype=torch.float32):
		self.device = device
		self.clip_dtype = clip_dtype
		self._init_clip()
		self.model = self._load_model(model_path)
		print("CityAesthetics: Pipeline init ok") # debug

	def __call__(self, raw):
		emb = self.get_clip_emb(raw)
		return self.get_model_pred(self.model, emb)

	def get_model_pred(self, model, emb):
		with torch.no_grad():
			pred = model(emb)
		return float(pred.detach().cpu().squeeze(0))

	def get_clip_emb(self, raw):
		img = self.proc(
			images = raw,
			return_tensors = "pt"
		)["pixel_values"].to(self.clip_dtype).to(self.device)
		with torch.no_grad():
			emb = self.clip(pixel_values=img)
		return emb["image_embeds"].detach().to(torch.float32)

	def _init_clip(self):
		self.proc = CLIPImageProcessor.from_pretrained(self.clip_ver)
		self.clip = CLIPVisionModelWithProjection.from_pretrained(
			self.clip_ver,
			device_map  = self.device,
			torch_dtype = self.clip_dtype,
		)

	def _load_model(self, path):
		sd = load_file(path)
		assert tuple(sd["up.0.weight"].shape) == (1024, 768) # only allow CLIP ver
		model = AestheticPredictorModel()
		model.eval()
		model.load_state_dict(sd)
		model.to(self.device)
		return model

class CityAestheticsMultiModelPipeline(CityAestheticsPipeline):
	"""
	Demo multi-model pipeline for [image=>score] prediction
		Accepts a list of model paths on initialization.
		Resulting object can be called directly with a PIL image as the input.
		Returns a dict with the model name as key and the score [0.0;1.0] as a value.
	"""
	def __init__(self, model_paths, device="cpu", clip_dtype=torch.float32):
		self.device = device
		self.clip_dtype = clip_dtype
		self._init_clip()
		self.models = {}
		for path in model_paths:
			name = os.path.splitext(os.path.basename(path))[0]
			self.models[name] = self._load_model(path)
		print("CityAesthetics: Pipeline init ok") # debug

	def __call__(self, raw):
		emb = self.get_clip_emb(raw)
		out = {}
		for name, model in self.models.items():
			pred = model(emb)
			out[name] = self.get_model_pred(model, emb)
		return out

def get_model_path(name, repo, token=True):
	"""
	Returns local model path or falls back to HF hub if required.
	"""
	fname = f"{name}.safetensors"

	# local path: [models/AesPred-Anime-v1.8.safetensors]
	path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"models")
	if os.path.isfile(os.path.join(path, fname)):
		print("CityAesthetics: Using local model")
		return os.path.join(path, fname)

	# huggingface hub fallback
	print("CityAesthetics: Using HF Hub model")
	return str(hf_hub_download(
		token    = token,
		repo_id  = repo,
		filename = fname,
	))
