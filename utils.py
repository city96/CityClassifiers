import os
import json
import yaml
import torch
import argparse
from tqdm import tqdm
from safetensors.torch import save_file

LOSS_MEMORY = 500
LOG_EVERY_N = 100
SAVE_FOLDER = "models"

def get_embed_params(ver):
	if ver == "CLIP":
		# CLIPVisionModelWithProjection
		#  openai/clip-vit-large-patch14
		return {
			"features" :  768,
			"hidden"   : 1024,
		}
	elif ver == "META":
		# open_clip
		#  metaclip_fullcc | ViT-H-14-quickgelu
		print("META ver. was only meant for testing!")
		return {
			"features" : 1024,
			"hidden"   : 1280,
		}
	else:
		raise ValueError(f"Unknown model '{ver}'")

def parse_args():
	parser = argparse.ArgumentParser(description="Train aesthetic predictor")
	parser.add_argument("--config", required=True, help="Training config")
	parser.add_argument('--resume', help="Checkpoint to resume from")
	parser.add_argument('--images', action=argparse.BooleanOptionalAction, default=False, help="Live process images")
	parser.add_argument("--nsave", type=int, default=0, help="Save model periodically")
	args = parser.parse_args()
	if not os.path.isfile(args.config):
		parser.error(f"Can't find config file '{args.config}'")
	args = get_training_args(args)
	return args

def get_training_args(args):
	with open(args.config) as f:
		conf = yaml.safe_load(f)
	train_conf = conf.get("train", {})
	args.lr    = train_conf.get(   "lr", 1e-6)
	args.steps = train_conf.get("steps", 100000)
	args.batch = train_conf.get("batch", 1)
	args.cosine= train_conf.get("cosine", True)

	assert "model" in conf.keys(), "Model config not optional!"
	args.base = conf["model"].get("base", "unknown")
	args.rev  = conf["model"].get("rev", "v0.0")
	args.arch = conf["model"].get("arch", None)
	args.clip = conf["model"].get("clip", "CLIP")
	args.name = f"{args.base}-{args.rev}"
	assert args.arch in ["score", "class"], f"Unknown arch '{args.arch}'"
	assert args.clip in ["CLIP", "META"], f"Unknown CLIP '{args.clip}'"

	labels = conf.get("labels", {})
	if args.arch == "class" and labels:
		args.labels = {str(k):v.get("name", str(k)) for k,v in labels.items()}
		args.num_labels = max([int(x) for x in labels.keys()])+1
		weights = [1.0 for _ in range(args.num_labels)]
		for k in labels.keys():
			weights[k] = labels[k].get("loss", 1.0)
		args.weights = weights
	else:
		args.num_labels = 1
		args.labels  = None
		args.weights = None
	return args

def write_config(args):
	conf = {
		"name"   : args.base,
		"rev"    : args.rev,
		"arch"   : args.arch,
		"labels" : args.labels,
	}
	conf["model_params"] = get_embed_params(args.clip)
	conf["model_params"]["outputs"] = args.num_labels

	os.makedirs(SAVE_FOLDER, exist_ok=True)
	with open(f"{SAVE_FOLDER}/{args.name}.config.json", "w") as f:
		f.write(json.dumps(conf, indent=2))

class ModelWrapper:
	def __init__(self, name, model, optimizer, criterion, scheduler, device="cpu", evals=[None,None], stdout=True):
		self.name   = name
		self.losses = []

		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.scheduler = scheduler

		self.device = device
		self.eval_src = evals.get("emb")
		self.eval_dst = evals.get("val")

		os.makedirs(SAVE_FOLDER, exist_ok=True)
		self.csvlog = open(f"{SAVE_FOLDER}/{self.name}.csv", "w")
		self.stdout = stdout

	def log_step(self, loss, step=None):
		self.losses.append(loss)
		step = step or len(self.losses)
		if step % LOG_EVERY_N == 0:
			self.log_main(step)

	def log_main(self, step=None):
		lr = float(self.scheduler.get_last_lr()[0])
		avg = sum(self.losses[-LOSS_MEMORY:])/LOSS_MEMORY
		evl, pred = self.eval_model()
		if self.stdout:
			if pred.shape[1] == 1:
				pred = f"{int(pred*100):03}/100"
			else:
				pred = ','.join([f"{x}:{int(pred[x][x]*100):03}%" for x in range(pred.shape[0])])
			tqdm.write(f"{str(step):<10} {avg:.4e}|{evl:.4e} @ {lr:.4e} = {pred}")
		if self.csvlog:
			self.csvlog.write(f"{step},{avg},{evl},{lr}\n")
			self.csvlog.flush()

	def eval_model(self):
		with torch.cuda.amp.autocast():
			with torch.no_grad():
				pred = self.model(self.eval_src.to(self.device))
				# tqdm.write(str(self.eval_dst))
				# tqdm.write(str(pred))
				loss = self.criterion(pred, self.eval_dst.to(self.device))
				# tqdm.write(str(loss))
		return loss, pred

	def save_model(self, step=None, epoch=None):
		step = step or len(self.losses)
		if epoch is None and step >= 10**6:
			epoch = f"_e{round(step/10**6,2)}M"
		elif epoch is None:
			epoch = f"_e{round(step/10**3)}K"
		output_name = f"./{SAVE_FOLDER}/{self.name}{epoch}"
		save_file(self.model.state_dict(), f"{output_name}.safetensors")
		torch.save(self.optimizer.state_dict(), f"{output_name}.optim.pth")

	def close(self):
		self.csvlog.close()
