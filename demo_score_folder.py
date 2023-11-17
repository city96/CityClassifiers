import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from shutil import copyfile

from inference import CityAestheticsPipeline

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

def parse_args():
	parser = argparse.ArgumentParser(description="Test model by running it on an entire folder")
	parser.add_argument('--src', default="test", help="Folder with images to score")
	parser.add_argument('--dst', default="pass", help="Folder with images that fit the score threshold")
	parser.add_argument('--max', type=int, default=100, help="Upper limit for score")
	parser.add_argument('--min', type=int, default=  0, help="Lower limit for score")
	parser.add_argument('--model', required=True, help="Model file to use")
	parser.add_argument("--arch", choices=["Aesthetic"], default="Aesthetic", help="Model type")
	parser.add_argument('--copy', action=argparse.BooleanOptionalAction, help="Copy files to the dst folder")
	parser.add_argument('--keep', action=argparse.BooleanOptionalAction, help="Keep original folder structure")
	return parser.parse_args()

def process_file(pipeline, src_path, dst_path):
	pred = pipeline(Image.open(src_path))
	pred = int(pred * 100) # [float]=>[int](percentage)

	tqdm.write(f" {pred:>3}% [{os.path.basename(src_path)}]")
	if args.min <= pred <= args.max:
		if dst_path: copyfile(src_path, dst_path)

def process_folder(pipeline, src_root, dst_root):
	dst_folders = [] # avoid excessive mkdir
	for path, _, files in os.walk(src_root):
		for fname in files:
			dst_path = None
			if args.copy:
				dst_dir = dst_root
				src_rel = os.path.relpath(path, src_root)
				if args.keep and src_rel != ".":
					dst_dir = os.path.join(dst_root, src_rel)
				if dst_dir not in dst_folders:
					os.makedirs(dst_dir, exist_ok=True)
					dst_folders.append(dst_dir)
				dst_path = os.path.join(dst_dir, fname)
			src_path = os.path.join(path, fname)
			if os.path.splitext(fname)[1] not in IMAGE_EXTS: continue
			process_file(pipeline, src_path, dst_path)
			# try: process_file(pipeline, src_path, dst_path)
			# except: pass # e.g. for skipping file errors

if __name__ == "__main__":
	args = parse_args()

	os.makedirs(args.dst, exist_ok=True)
	print(f"Aesthetic Predictor using model {os.path.basename(args.model)}")

	if args.arch == "Aesthetic":
		pipeline_args = {}
		if torch.cuda.is_available():
			pipeline_args["device"] = "cuda"
			pipeline_args["clip_dtype"] = torch.float16
		pipeline = CityAestheticsPipeline(args.model, **pipeline_args)
	else:
		raise ValueError(f"Unknown model architecture '{args.arch}'")

	process_folder(pipeline, args.src, args.dst)
