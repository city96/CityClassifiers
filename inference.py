import os
import json
import torch
import torchvision.transforms as TF
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from model import PredictorModel

class CityAestheticsPipeline:
	"""
	Demo model pipeline for [image=>score] prediction
		Accepts a single model path on initialization.
		Resulting object can be called directly with a PIL image as the input
		Returns a single float value with the predicted score [0.0;1.0].
	"""
	clip_ver = "openai/clip-vit-large-patch14-336"
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
		model = PredictorModel(outputs=1)
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

class CityClassifierPipeline:
	"""
	Demo model pipeline for [image=>label] prediction
		Accepts a single model path and (optionally) a JSON file on initialization.
		Resulting object can be called directly with a PIL image as the input
		Returns a single float value with the predicted score [0.0;1.0].
	"""
	clip_ver = "openai/clip-vit-large-patch14-336"
	def __init__(self, model_path, config_path=None, device="cpu", clip_dtype=torch.float32):
		self.device = device
		self.clip_dtype = clip_dtype
		self._init_clip()

		self.labels, model_args = self._load_config(config_path)
		self.model = self._load_model(model_path, model_args)

		print("CityClassifier: Pipeline init ok") # debug

	def __call__(self, raw, default=True, tiling=True, tile_strat="mean"):
		emb = self.get_clip_emb(raw, tiling=tiling)
		pred = self.get_model_pred(self.model, emb)
		return self.format_pred(
			pred,
			labels = self.labels,
			drop = [] if default else [0],
			ts = tile_strat if tiling else "raw",
		)

	def format_pred(self, pred, labels, drop=[], ts="mean"):
		# recombine strategy
		if   ts == "mean"  : vp = lambda x: float(torch.mean(x))
		elif ts == "median": vp = lambda x: float(torch.median(x))
		elif ts == "max"   : vp = lambda x: float(torch.max(x))
		elif ts == "min"   : vp = lambda x: float(torch.min(x))
		elif ts == "raw"   : vp = lambda x: float(x)
		else: raise NotImplementedError(f"CityClassifier: Invalid combine strategy '{ts}'!")
		# combine pred w/ labels
		out = {}
		for k in range(len(pred)):
			if k in drop: continue
			key = labels.get(str(k), str(k))
			out[key] = vp(pred[k])
		return out

	def get_model_pred(self, model, emb):
		with torch.no_grad():
			pred = model(emb)
		pred = pred.detach().cpu()
		return [pred[:, x] for x in range(pred.shape[1])] # split

	def get_clip_emb(self, raw, tiling=False):
		if tiling and min(raw.size) > self.size*2:
			if max(raw.size)>1536:
				raw = TF.functional.resize(raw, 1536)
			raw = TF.functional.five_crop(raw, self.size*2)
		img = self.proc(
			images = raw,
			return_tensors = "pt"
		)["pixel_values"].to(self.clip_dtype).to(self.device)
		with torch.no_grad():
			emb = self.clip(pixel_values=img)
		return emb["image_embeds"].detach().to(torch.float32)

	def _init_clip(self):
		self.proc = CLIPImageProcessor.from_pretrained(self.clip_ver)
		self.size = self.proc.size.get("shortest_edge", 256)
		self.clip = CLIPVisionModelWithProjection.from_pretrained(
			self.clip_ver,
			device_map  = self.device,
			torch_dtype = self.clip_dtype,
		)

	def _load_model(self, path, args=None):
		sd = load_file(path)
		assert tuple(sd["up.0.weight"].shape) == (1024, 768) # only allow CLIP ver
		args = args or { # infer from model
			"outputs" : int(sd["down.5.bias"].shape[0])
		}
		model = PredictorModel(**args)
		model.eval()
		model.load_state_dict(sd)
		model.to(self.device)
		return model

	def _load_config(self, path):
		if not path or not os.path.isfile(path):
			return ({},None)

		with open(path) as f:
			data = json.loads(f.read())
		return (
			data.get("labels", {}),
			data.get("model_params", {}),
		)

class CityClassifierMultiModelPipeline(CityClassifierPipeline):
	"""
	Demo model pipeline for [image=>label] prediction
		Accepts a list of model paths on initialization.
		A matching list of JSON files can also be passed in the same order.
		Resulting object can be called directly with a PIL image as the input
		Returns a single float value with the predicted score [0.0;1.0].
	"""
	def __init__(self, model_paths, config_paths=[], device="cpu", clip_dtype=torch.float32):
		self.device = device
		self.clip_dtype = clip_dtype
		self._init_clip()
		self.models = {}
		self.labels = {}
		assert len(model_paths) == len(config_paths) or not config_paths, "CityClassifier: Model and config paths must match!"
		for k in range(len(model_paths)):
			name = os.path.splitext(os.path.basename(model_paths[k]))[0] # TODO: read from config
			self.labels[name], model_args = self._load_config(config_paths[k] if config_paths else None)
			self.models[name] = self._load_model(model_paths[k], model_args)
			
		print("CityClassifier: Pipeline init ok") # debug

	def __call__(self, raw, default=True, tiling=True, tile_strat="mean"):
		emb = self.get_clip_emb(raw, tiling=tiling)
		out = {}
		for name, model in self.models.items():
			pred = self.get_model_pred(model, emb)
			out[name] = self.format_pred(
				pred,
				labels = self.labels[name],
				drop = [] if default else [0],
				ts = tile_strat if tiling else "raw",
			)
		if len(out.values()) == 1: return list(out.values())[0] # GRADIO HOTFIX
		return list(out.values())

def get_model_path(name, repo, token=True, extension="safetensors", local=False):
	"""
	Returns local model path or falls back to HF hub if required.
	"""
	fname = f"{name}.{extension}"

	# local path: [models/AesPred-Anime-v1.8.safetensors]
	path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"models")
	if os.path.isfile(os.path.join(path, fname)):
		print(f"Using local model for '{fname}'")
		return os.path.join(path, fname)

	if local: raise OSError(f"Can't find local model '{fname}'!")

	# huggingface hub fallback
	print(f"Using HF Hub model for '{fname}'")
	return str(hf_hub_download(
		token    = token,
		repo_id  = repo,
		filename = fname,
	))
