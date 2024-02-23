# Data


In our work, we consider two datasets:

- Million Reddit User Dataset (MUD) (https://arxiv.org/abs/2105.07263, https://docs.google.com/forms/d/e/1FAIpQLSesc-0HI2DRYjFqlpPo2hTh9OJ53jtWjYQiIfAtmzSVUCxiLA/viewform)

- The Enron Email Corpus (https://www.cs.cmu.edu/~./enron/, https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

Each folder contains the preprocessing logic for each dataset (in each, see `generate_dataset.sh`)


Alternatively, for Enron, you can directly download the preprocesses training data (see `download_training_dataset.sh`)
This downloads a data folder (`2023-06-22-23.15.35`), which contains:

- `2023-06-22-23.15.35/max_len_50_min_score_None`, the preprocessed training dataset
- `2023-06-22-23.15.35/t5_max_len_50_roberta_determined_min_score_None`, the dataset but retokenized with the t5 tokenizer (for our t5 STRAP baselines)


The enron folder also contains the holdout splits used for evaluating attribute and authorship transfer.

