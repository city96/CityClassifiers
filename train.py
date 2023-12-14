import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from dataset import EmbeddingDataset, ImageDataset
from utils import ModelWrapper, get_embed_params, parse_args, write_config
from model import PredictorModel

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

TARGET_DEV = "cuda"

if __name__ == "__main__":
	args = parse_args()

	if args.images: dataset = ImageDataset(args.clip, mode=args.arch)
	else: dataset = EmbeddingDataset(args.clip, mode=args.arch, preload=True)

	loader = DataLoader(
		dataset,
		batch_size  = args.batch,
		shuffle     = True,
		drop_last   = True,
		pin_memory  = False,
		num_workers = 0,
		# num_workers=4, # doesn't work w/ --image
		# persistent_workers=True,
	)

	if args.arch == "score":
		criterion = torch.nn.L1Loss()
		model = PredictorModel(
			outputs = 1,
			**get_embed_params(args.clip)
		)
	elif args.arch == "class":
		args.num_labels = args.num_labels or dataset.num_labels
		assert args.num_labels == dataset.num_labels, "Label count mismatch!"
		weights = None
		if args.weights:
			weights = torch.tensor(args.weights, device=TARGET_DEV)
			print(f"Class weights: '{args.weights}'")
		criterion = torch.nn.CrossEntropyLoss(weights)
		model = PredictorModel(
			outputs = args.num_labels,
			**get_embed_params(args.clip)
		)
	else:
		raise ValueError(f"Unknown model architecture '{args.arch}'")

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

	write_config(args) # model config file
	wrapper = ModelWrapper( # model wrapper for saving/eval/etc
		name      = args.name,
		model     = model,
		device    = TARGET_DEV,
		dataset   = dataset,
		criterion = criterion,
		optimizer = optimizer,
		scheduler = scheduler,
	)

	progress = tqdm(total=args.steps)
	while progress.n < args.steps:
		for batch in loader:
			emb = batch.get("emb").to(TARGET_DEV)

			if args.arch == "score":
				val = batch.get("val").to(TARGET_DEV)
			elif args.arch == "class":
				val = torch.zeros(args.num_labels, device=TARGET_DEV)
				val[batch.get("raw")] = 1.0 # expand classes
				val = val.unsqueeze(0).repeat(emb.shape[0], 1)

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
			wrapper.log_point(loss.data.item(), batch.get("index"))
			if args.nsave > 0 and progress.n % (args.nsave + args.nsave%args.batch) == 0:
				wrapper.save_model(step=progress.n)
			if progress.n >= args.steps:
				break
	progress.close()
	wrapper.save_model(epoch="") # final save
	wrapper.enum_point() # outliers
	wrapper.close()
