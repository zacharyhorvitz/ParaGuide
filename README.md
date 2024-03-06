<h2 align="center">
  ParaGuide
</h2>
<h3 align="center">
:electric_plug: :robot: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer  :robot: :electric_plug:
</h3>

Our paper is available [here](https://arxiv.org/abs/2308.15459).

### Structure

- [demo/](demo/) contains demo code, demonstrating how to run the full ParaGuide approach to transform texts from formal &#8594; informal, or to match exemplars.

- [training/](training/) contains the logic for training a paraphrase-conditioned text diffusion model.

- [inference/](inference/) contains our code for generating inferences with ParaGuide (once you have preprocessed/paraphrased your data)

- [data/](data/) contains the logic for generating reddit and enron (paraphrase, original text) data.

- [baselines/](baselines/) contains our implementations of each baseline.

- [evaluations](evaluations/) contains our automatic evaluation [automatic](`evaluations/automatic_evals/`) and [human](evaluations/human_evals/) eval data and code.

### Getting Started

#### Installation

In your python environment (>=3.8), you can install dependencies via the requirements file:

```bash
pip install -r requirements.txt
```

#### Models and Data Artifacts

Our models and data are available for download [here](https://drive.google.com/drive/folders/1Pz8IcM3TWQOHK6UfC7XqlBd5TqR3Vo3b).

We also provided corresponding scripts:

- **Models**: [models/download.sh](models/download.sh)
- **Data**: [data/enron/download_training_dataset.sh](data/enron/download_training_dataset.sh)

### Demo

We recommend first checking out [demo/generate_examples.py](demo/generate_examples.py), which demonstrates ParaGuide inference logic!








