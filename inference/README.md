# Inference

This folder contains out logic for generating ParaGuide inferences. 

Please note that this logic expects preprocessed data (See ``data/enron``), with **(paraphrase, original text)** pairs. 
To run the full ParaGuide pipeline without preprocessing, see ``demo/``.

``example_inference.sh`` contains an example invocation of the inference script.

``inference.py`` contains our inference logic for both **Enron attribute transfer** (formality or sentiment) and **authorship transfer** (with style embeddings).

``classifiers.py`` contains our utilities for loading and using different guidance models.

``diffusion_utils.py`` implements our diffusion inference utilities.

``inference_utils.py`` implements our general inference/configuration utilities.


### TODO:
- Include LUAR guidance support (https://aclanthology.org/2021.emnlp-main.70/)
