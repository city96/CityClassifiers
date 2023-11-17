import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from dataset import EmbeddingDataset
from utils import ModelWrapper, get_embed_params
from model import AestheticPredictorModel

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

TARGET_DEV = "cuda"

def parse_args():
	parser = argparse.ArgumentParser(description="Train aesthetic predictor")
	parser.add_argument("-s", "--steps", type=int, default=100000, help="No. of training steps")
	parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
	parser.add_argument("-n", "--nsave", type=int, default=0, help="Save model/sample periodically")
	parser.add_argument('--lr', default="7e-6", help="Learning rate")
	parser.add_argument('--rev', default="Anime-v1.9-rc1", help="Revision/log ID")
	parser.add_argument("--clip", choices=["CLIP", "META"], default="CLIP", help="Embedding type")
	parser.add_argument("--arch", choices=["Aesthetic"], default="Aesthetic", help="Model type")
	parser.add_argument('--resume', help="Checkpoint to resume from")
	parser.add_argument('--cosine', action=argparse.BooleanOptionalAction, default=True, help="Use cosine scheduler")
	args = parser.parse_args()
	try:
		float(args.lr)
	except ValueError:
		parser.error("--lr must be a valid float eg. 0.001 or 1e-3")
	return args

if __name__ == "__main__":
	args = parse_args()

	dataset = EmbeddingDataset(args.clip, preload=True)
	loader = DataLoader(
		dataset,
		batch_size  = args.batch,
		shuffle     = True,
		drop_last   = True,
		pin_memory  = False,
		num_workers = 0,
		# num_workers=4,
		# persistent_workers=True,
	)

	if args.arch == "Aesthetic":
		model = AestheticPredictorModel(**get_embed_params(args.clip))
		name  = f"CityAesthetics-{args.rev}"
		#name = f"AesPred-{args.ver}-{args.rev}", # for testing different CLIP versions
	else:
		raise ValueError(f"Unknown model architecture '{args.arch}'")

	criterion = torch.nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
	scheduler = None
	if args.cosine:
		print("Using CosineAnnealingLR")
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max = int(args.steps/args.batch),
		)
	else:
		print("Using LinearLR")
		scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor = 0.1,
			end_factor   = 1.0,
			total_iters  = int(5000/args.batch),
		)

	if args.resume:
		model.load_state_dict(load_file(args.resume))
		model.to(TARGET_DEV)
		optimizer.load_state_dict(torch.load(
			f"{os.path.splitext(args.resume)[0]}.optim.pth"
		))
		optimizer.param_groups[0]['lr'] = scheduler.base_lrs[0]
	else:
		model.to(TARGET_DEV)

	wrapper = ModelWrapper( # model wrapper for saving/eval/etc
		name      = name,
		model     = model,
		evals     = dataset.get_eval(),
		device    = TARGET_DEV,
		criterion = criterion,
		optimizer = optimizer,
		scheduler = scheduler,
	)

	progress = tqdm(total=args.steps)
	while progress.n < args.steps:
		for batch in loader:
			emb = batch.get("emb").to(TARGET_DEV)
			val = batch.get("val").to(TARGET_DEV)
			with torch.cuda.amp.autocast():
				y_pred = model(emb) # forward
				loss = criterion(y_pred, val) # loss

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			# eval/save
			progress.update(args.batch)
			wrapper.log_step(loss.data.item(), progress.n)
			if args.nsave > 0 and progress.n % (args.nsave + args.nsave%args.batch) == 0:
				wrapper.save_model(step=progress.n)
			if progress.n >= args.steps:
				break
	progress.close()
	wrapper.save_model(epoch="") # final save
	wrapper.close()
