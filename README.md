# Amazon Recommendation System

Production-style recommender-system project built from the original
`AmazonRecommendationColab.ipynb` notebook. The notebook is preserved for
reference, while the reusable workflow now lives in Python modules, scripts,
tests, and GitHub Actions.

## What this project does

This project builds product recommendations from Amazon All Beauty reviews. It
uses collaborative filtering ideas from the notebook, but keeps the core model
logic implemented directly with NumPy/Pandas:

- Popularity baseline with Bayesian average ratings
- User-based KNN collaborative filtering
- Item-based KNN collaborative filtering
- Low-rank SVD recommender

The project does not require Spark, Surprise, or scikit-learn for the core
recommenders.

## Dataset

The data comes from the UCSD/McAuley Amazon Review Data 2018 dataset:

- Dataset page: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- Review file used here: All Beauty 5-core subset
- Optional metadata file: All Beauty product metadata

The dataset page describes the 2018 release as an updated Amazon review dataset
with reviews, ratings, helpfulness votes, product metadata, and product links.
The All Beauty 5-core subset contains 5,269 reviews, which keeps local
experiments quick and reproducible.

Raw data is intentionally ignored by git. Download it locally with:

```bash
python scripts/download_data.py
```

## Project Structure

```text
.
|-- AmazonRecommendationColab.ipynb   # original notebook, preserved
|-- scripts/
|   |-- download_data.py              # reproducible data download
|   `-- train.py                      # train/evaluate recommendation models
|-- src/amazon_recommender/
|   |-- data.py                       # loading, cleaning, ID encoding
|   |-- metrics.py                    # MAE/RMSE evaluation
|   |-- models.py                     # custom recommender implementations
|   `-- reporting.py                  # readable recommendation outputs
|-- tests/                            # unit tests
`-- .github/workflows/tests.yml       # CI
```

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download the dataset:

```bash
python scripts/download_data.py
```

Train and evaluate the recommenders:

```bash
python scripts/train.py
```

Run tests:

```bash
python -m pytest
```

## Baseline Results

After removing duplicate reviewer/product pairs, the current pipeline evaluates
4,092 ratings from 991 users and 85 products with an 80/20 random split.

| Model | MAE | RMSE | Coverage |
| --- | ---: | ---: | ---: |
| Popularity baseline | 0.3344 | 0.6779 | 1.0000 |
| User KNN | 0.0926 | 0.3838 | 1.0000 |
| Item KNN | 0.0952 | 0.3154 | 1.0000 |
| SVD | 0.0967 | 0.3899 | 1.0000 |

The training script also prints sample top-N SVD product recommendations for a
user.

## Notes

- `data/`, `reports/`, and model artifacts are ignored so large generated files
  do not get committed.
- The original Colab notebook remains unchanged.
- The production path is designed for repeatable local runs and CI validation.

## Original Report

[AmazonRecommendation_FinalReport.pdf](https://github.com/Ayush210895/Amazon-Recommendation-System/files/9805583/AmazonRecommendation_FinalReport.pdf)
