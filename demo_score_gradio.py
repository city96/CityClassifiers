import os
import gradio as gr

from inference import CityAestheticsMultiModelPipeline, get_model_path

TOKEN  = os.environ.get("HFS_TOKEN")
HFREPO = "City96/CityAesthetics"
MODELS = [
	"CityAesthetics-Anime-v1.8",
]
article = """\
# About

This is the live demo for the CityAesthetics class of predictors.

For more information, you can check out the [Huggingface Hub](https://huggingface.co/city96/CityAesthetics) or [GitHub page](https://github.com/city96/CityClassifiers).

## CityAesthetics-Anime

This flavor is optimized for scoring anime images with at least one subject present.

### Intentional biases:

- Completely negative towards real life photos (ideal score of 0%)
- Strongly Negative towards text (subtitles, memes, etc) and manga panels
- Fairly negative towards 3D and to some extent 2.5D images
- Negative towards western cartoons and stylized images (chibi, parody)

### Expected output scores:

- Non-anime images should always score below 20%
- Sketches/rough lineart/oekaki get around 20-40%
- Flat shading/TV anime gets around 40-50%
- Above 50% is mostly scored based on my personal style preferences

### Issues:

- Tends to filter male characters.
- Requires at least 1 subject, won't work for scenery/landscapes.
- Noticeable positive bias towards anime characters with animal ears.
- Hit-or-miss with AI generated images due to style/quality not being correlated.
"""

pipeline = CityAestheticsMultiModelPipeline(
	[get_model_path(x, HFREPO, TOKEN) for x in MODELS],
)
gr.Interface(
	fn      = pipeline,
	title   = "CityAesthetics demo",
	article = article,
	inputs  = gr.Image(label="Input image", type="pil"),
	outputs = gr.Label(label="Model prediction", show_label=False),
	examples = "./examples" if os.path.isdir("./examples") else None,
	allow_flagging = "never",
	analytics_enabled = False,
).launch()
