# Gaussian Naive Bayes Classifier from scratch

## Objective

The goal of this project is to implement Gaussian Naive Bayes from scratch using NumPy. Learn how to fit the model, compute log-likelihoods, apply numerical stability tricks like log-sum-exp, and build a vectorized classifier step-by-step.

## Blog post

You can find the blog post series here:

- [Gaussian Naive Bayes: Part 1 — Introduction and Bayes Theorem Refresher](https://stefannieuwenhuis.github.io/2025/05/19/math-behind-naive-bayes-part1.html)
- [Gaussian Naive Bayes: Part 2 — Mathematical Deep Dive and Optimization](https://stefannieuwenhuis.github.io/2025/05/19/math-behind-naive-bayes-part2.html)
- [Gaussian Naive Bayes: Part 3 — From Theory to Practice with Python](https://stefannieuwenhuis.github.io/2025/05/24/math-behind-naive-bayes-part3.html)

## Iris Flower Species dataset

![Iris Flower Species](https://stefannieuwenhuis.github.io/assets/images/iris_species_with_labels.png)

The Iris flower data set or Fisher’s Iris data set is a multivariate data set used and made famous by the British statistician and biologist Ronald Fisher in his 1936 paper [The Use of Multiple Measurements in Taxonomic Problems](https://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf) as an example of linear discriminant analysis.

The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris Virginica and Iris Versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

<small>– Source: [Wikipedia: Iris Flower Data Set](https://en.wikipedia.org/wiki/Iris_flower_data_set)</small>


In `/notebooks/01_eda.ipynb`, we explore the dataset and get preliminary insights.

## Folder Structure

```
.
├── src/
|   ├── model/
|   |   └── gaussian_naive_bayes.py     # Core GaussianNB implementation
├── tests/                              # Unit tests
├── notebooks/
│   └── 01_eda.ipynb                    # Exploratory Data Analysis
│   └── 02_implementation.ipynb         # Gaussian Naive Bayes Classifier implementation
│   └── 03_evaluation.ipynb             # Gaussian Naive Bayes Classifier model evaluation
├── Makefile                            # Automation: env, jupyter, test
├── requirements.txt                    # Runtime dependencies
├── LICENSE.md                          # MIT LICENSE file
├── README.md                           # README with details
```

## Setup

```shell
# Create and activate a virtual environment, and install dependencies
make env

# Run Jupyterlab in /notebooks/
make jupyter 

# Run unit tests from /tests/ directory
make test
```