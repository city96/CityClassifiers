import os
import torch
import gradio as gr

from inference import CityClassifierMultiModelPipeline, get_model_path

TOKEN  = os.environ.get("HFS_TOKEN")
HFREPO = "City96/AnimeClassifiers"
MODELS = [
	"CCAnime-ChromaticAberration-v1.16",
]
article = """\
# About

These are classifiers meant to work with anime images.

For more information, you can check out the [Huggingface Hub](https://huggingface.co/city96/AnimeClassifiers) or [GitHub page](https://github.com/city96/CityClassifiers).
"""
info_default="""\
Include default class (unknown/negative) in output results.
"""
info_tiling = """\
Divide the image into parts and run classifier on each part separately.
Greatly improves accuracy but slows down inference.
"""
info_tiling_combine = """\
How to combine the confidence scores of the different tiles.
Mean averages confidence over all tiles. Median takes the value in the middle.
Max/min take the score from the tile with the highest/lowest confidence respectively, but can results in multiple labels having very high/very low confidence scores.
"""

pipeline_args = {}
if torch.cuda.is_available():
	pipeline_args.update({
		"device"     : "cuda",
		"clip_dtype" : torch.float16,
	})

pipeline = CityClassifierMultiModelPipeline(
	model_paths = [get_model_path(x, HFREPO, TOKEN) for x in MODELS],
	config_paths = [get_model_path(x, HFREPO, TOKEN, extension="config.json") for x in MODELS],
	**pipeline_args,
)
gr.Interface(
	fn      = pipeline,
	title   = "Anime Classifiers - demo",
	article = article,
	inputs  = [
		gr.Image(label="Input image", type="pil"),
		gr.Checkbox(label="Include default", value=True, info=info_default),
		gr.Checkbox(label="Tiling", value=True, info=info_tiling),
		gr.Dropdown(
			label   = "Tiling combine strategy",
			choices = ["mean", "median", "max", "min"],
			value = "mean",
			type = "value",
			info = info_tiling_combine,
		)
	],
	outputs = [gr.Label(label=x) for x in MODELS],
	examples = "./examples" if os.path.isdir("./examples") else None,
	allow_flagging = "never",
	analytics_enabled = False,
).launch()
