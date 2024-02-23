# Training


Logic for training a ParaGuide model. See ``train.sh`` for a training template.

In our work, we fine-tune the original SSD-LM diffusion model (https://huggingface.co/xhan77/ssdlm), with our modified approach/architecture.

To fine-tune our Enron Model, you will first need to download our checkpoint (See ``models/download.sh``), and preprocess your data (See ``data/``).

``train.py`` contains our training code.
