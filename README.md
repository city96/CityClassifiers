# CityClassifiers

Code for my collection of predictors/classifiers/etc

## Architecture

The base model itself is fairly simple. It takes embeddings from a CLIP model (in this case, `openai/clip-vit-large-patch14-336`) and expands them to 1024 dimensions. From there, a single block with residuals is followed by a few linear layers which converge down to the final output.

For the predictor model, the final output goes through `nn.Tanh`. For the classifier, this is `nn.Softmax` instead.

# Classifiers

[Live Demos](https://huggingface.co/spaces/city96/AnimeClassifiers-demo) | [Model downloads](https://huggingface.co/city96/AnimeClassifiers)

These are models that predict whether a concept is present in an image. The performance on high resolution images isn't very good, especially when detecting subtle image effects such as noise. This is due to CLIP using a fairly low resolution (336x336/224x224).

To combat this, tiling is used at inference time. The input image is first downscaled to 1536 (shortest edge - See `TF.functional.resize`), then 5 separate areas are selected (double the res of the CLIP preprocessor. 4 corners + center - See `TF.functional.five_crop`). This helps as the downscale factor isn't nearly as drastic as passing the entire image to CLIP. As a bonus, it also avoids the issues with odd aspect ratios requiring cropping or letterboxing to work.

![Tiling](https://github.com/city96/CityClassifiers/assets/125218114/66a30048-93ce-4c00-befc-0d986c84ec9f)

As for the training, it will be detailed in the sections below for the individual classifiers. At first, specialized models will be trained to a relatively high accuracy, building up a high quality but specific dataset in the process.

Then, these models will be used to split/sort each other's the datasets. The code will need to be updated to support one image being part of more than one class, but the final result should be a clean dataset where each target aspect acts as a "tag" rather than a class.

## Future/planned

- Unified (by joining the datasets of the other classifiers)
- Compression (jpg/webp/gif/dithering/etc)
- Noise

## ChromaticAberration - Anime

### Design goals

The goal was to detect [chromatic aberration](https://en.wikipedia.org/wiki/Chromatic_aberration?useskin=vector) in images.

For some odd reason, this effect has become a popular post processing effect to apply to images and drawings. While attempting to train an ESRGAN model, I noticed an odd halo around images and quickly figured out that this effect was the cause. This classifier aims to work as a base filter to remove such images from the dataset.

### Issues

- Seems to get confused by excessive HSV noise
- Triggers even if the effect is only applied to the background
- Sometimes triggers on rough linework/sketches (i.e. multiple semi-transparent lines overlapping)
- Low accuracy on 3D/2.5D with possible false positives.

### Training

The training settings can be found in the `config/CCAnime-ChromaticAberration-v1.yaml` file (7e-6 LR, cosine scheduler, 100K steps).

![loss](https://github.com/city96/CityClassifiers/assets/125218114/475f1241-2b4e-4fc9-bbcd-261b85b8b491)

![loss-eval](https://github.com/city96/CityClassifiers/assets/125218114/88d6f090-aa6f-42ad-9fd0-8c5d267fce5e)


Final dataset score distribution for v1.16:
```
3215 images in dataset.
0_reg       -  395 ||||
0_reg_booru - 1805 ||||||||||||||||||||||
1_chroma    -  515 ||||||
1_synthetic -  500 ||||||

Class ratios:
00 - 2200 |||||||||||||||||||||||||||
01 - 1015 ||||||||||||
```

Version history:

- v1.0 - Initial test model, dataset is fully synthetic (500 images). Effect added by shifting red/blue channel by a random amount using chaiNNer.
- v1.1 - Added 300 images tagged "chromatic_aberration" from gelbooru. Added first 1000 images from danbooru2021 as reg images
- v1.2 - Used the newly trained predictor to filter the existing datasets - found ~70 positives in the reg set and ~30 false positives in the target set.
- v1.3-v1.16 - Repeatedly ran predictor against various datasets, adding false positives/negatives back into the dataset, sometimes running against the training set to filter out misclassified images as the predictor got better. Added/removed images were manually checked (My eyes hurt).


# Predictors

## CityAesthetics - Anime

![Logo](https://github.com/city96/CityClassifiers/assets/125218114/0413003a-851d-42fc-b795-eae525b7b2e5)

[Live Demo](https://huggingface.co/spaces/city96/CityAesthetics-demo) | [Model download](https://huggingface.co/city96/CityAesthetics)

### Design goals

The goal was to create an aesthetic predictor that can work well on one specific type of image (in this case, anime) while filtering out everything else. To achieve this, the model was trained on a set of 3080 hand-scored images with multiple refinement steps, where false positives and negatives would be added to the training set with corrected scores after each test run.

This model focuses on as few false positives as possible. Only having one type of media seems to help with this, as predictors that attempt to do both real life and 2D images tend to produce false positives. If one were to have a mixed dataset with both types of images, then the simplest solution would be to use two separate aesthetic score models and a classifier to pick the appropriate one to use.

#### Intentional biases

- Completely negative towards real life photos (ideal score of 0%)
- Strongly Negative towards text (subtitles, memes, etc) and manga panels
- Fairly negative towards 3D and to some extent 2.5D images
- Negative towards western cartoons and stylized images (chibi, parody)

#### Issues

- Tends to filter male characters due to being underrepresented in the training set
- Requires at least 1 subject to be present in the image - doesn't work for scenery/landscapes
- Noticeable positive bias towards anime characters with animal ears
- Hit-or-miss with AI generated images due to style/quality not being correlated

#### Out-of-scope

- This model is not meant for moderation/live filtering/etc
- The demo code is not meant to work with large-scale datasets and is therefore only single-threaded. If you're working on something that requires an optimized version that can work on pre-computed CLIP embeddings for faster iteration, feel free to [contact me](mailto:city@eruruu.net).

### Usecases

The main usecase will be to provide baseline filtering on large datasets (i.e. a high pass filter). For this, the score brackets were decided as follows:

- <10% - Real life photos, noise, excessive text (subtitles, memes, etc)
- 10-20% - Manga panels, images with no subject, non-human subjects
- 20-40% - Sketches, oekaki, rough lineart (score depends on quality)
- 40-50% - Flat shading, TV anime screenshots, average images
- \>50% - "High quality" images based on my personal style preferences

The \>60% score range is intended to help pick out the "best" images from a dataset. One could use it to filter by score (i.e. using it as a band pass filter), but the scores above 50% are a lot more vague. Instead, I'd recommend sorting the dataset by score instead and setting a limit on the total number of images to select.

Top 100 images from a subset of danbooru2021 using the v1.7 model:

![AesPredv17_T100C](https://github.com/city96/CityClassifiers/assets/125218114/b7d8a167-a53a-46bb-8737-6c6c2a04f50f)

### Training

The training settings are initialized from the `config/CityAesthetics-v1.yaml` file (7e-6 LR, cosine scheduler, 100K steps).

![loss](https://github.com/city96/CityClassifiers/assets/125218114/611ae144-1390-48d3-988d-59a03c4a2f26)

Final dataset score distribution for v1.8:
```
3080 images in dataset.
  0 -   31 |
  1 -  162 |||||
  2 -  533 |||||||||||||||||
  3 -  675 |||||||||||||||||||||
  4 -  690 ||||||||||||||||||||||
  5 -  576 ||||||||||||||||||
  6 -  228 |||||||
  7 -   95 |||
  8 -   54 |
  9 -   29
 10 -    7
raw -    0
```

Version history:

- v1.0 - Initial test model with ~150 images to test viability
- v1.1 - Initialized top 5 score brackets with ~250 hand-picked images
- v1.2 - Manually scored ~2500 danbooru images for the main training set
- v1.3-v1.7 - Repeatedly ran the model against various datasets, adding the false negatives/positives to the training set to try and correct for various edgecases
- v1.8 - Added 3D and 2.5D images to the negative brackets to filter these as well

### Demo

A live demo can be accessed on [Huggingface](https://huggingface.co/spaces/city96/CityAesthetics-demo). The same demo can also be started locally by running `demo_score_gradio.py` after installing the requirements (`pip install -r requirements.txt`). Optionally, if a "models" folder with the correct files is present, then it will be used instead of huggingface.

`demo_score_folder.py` is a simple test script that can be used to recursively score all images in a folder, optionally copying the images between a set threshold to the output folder. Check --help for more info.
