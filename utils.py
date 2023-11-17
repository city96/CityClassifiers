import os
import torch
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
			tqdm.write(f"{str(step):<10} {avg:.4e}|{evl:.4e} @ {lr:.4e} = {int(pred*100):3.0f}/100")
		if self.csvlog:
			self.csvlog.write(f"{step},{avg},{evl},{lr}\n")
			self.csvlog.flush()

	def eval_model(self):
		with torch.no_grad():
			pred = self.model(self.eval_src.to(self.device))
			loss = self.criterion(pred, self.eval_dst.to(self.device))
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
