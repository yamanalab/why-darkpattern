<div align="center">

# Why is the User Interface a Dark Pattern? : Explainable Auto-Detection and its Analysis

[Yuki Yada](https://www.yyada.jp/), Tsuneo Matsumoto, Fuyuko Kido, Hayato Yamana

</div>

This repository provides the text-based dataset and experimental code for the paper: **Why is the User Interface a Dark Pattern? : Explainable Auto-Detection and its Analysis**

Accepted at **IEEE BigData 2023 (Poster)**

## Overview

Dark patterns are malicious user interface designs that lead users towards specific actions and have been a growing concern in recent years.

This research focused on the interpretable automatic detection of dark patterns, specifically on extracting the reasons why a user interface is determined to be a dark pattern.

First, we constructed an automatic detection model for dark patterns using BERT, based on a dataset obtained from prior research for dark patterns auto detection on E-Commerce sites. Next, we applied LIME and SHAP, which are Post-Hoc interpretation methods for machine learning models, to extract words that influence the determination of a dark pattern.

For more information, please check our paper. <!-- TODO: URL -->

## Setup

**Requirements:**

- python ^3.8
- poetry 1.2.1

You can setup project by running:

```bash
$ poetry install
```

Set PYTHONPATH to environment variable

```bash
$ export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## How to Run

To train and evaluate dark pattern auto detection model, please run:

```bash
$ python experiments/train.py
```

You can execute lime-based interpretation of dark pattern auto detection model by:

```bash
$ python experiments/explain_by_lime.py
```

For shap:

```bash
$ python experiments/explain_by_shap.py
```

## Experimental Result

### Performance Evaluation: Dark Pattern Auto Detection

|          Model           |     Accuracy     |       AUC        |     F1 score     |    Precision     |      Recall      |
| :----------------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
|   $\text{BERT}_{base}$   |      0.958       |      0.991       |      0.959       |      0.949       | $\mathbf{0.969}$ |
|  $\text{BERT}_{large}$   |      0.967       |      0.992       |      0.967       |      0.972       |      0.962       |
| $\text{RoBERTa}_{base}$  |      0.965       | $\mathbf{0.992}$ |      0.965       |      0.970       |      0.960       |
| $\text{RoBERTa}_{large}$ | $\mathbf{0.969}$ |      0.991       | $\mathbf{0.969}$ | $\mathbf{0.981}$ |      0.957       |

### Local Interpretation by LIME

We have applied LIME to all instances and visualized the importance scores (Saliency) by coloring them, as shown below:

<img width="440" alt="lime" src="https://github.com/yamanalab/why-darkpattern/assets/57289763/0f1427ee-b589-45a0-b1f1-69b5ad1287cc">

### Global Interpretation by SHAP

We applied SHAP to all dark pattern texts. We calculated the average score ranged from 0 to 1, from SHAP and extracted words in descending order based on their average scores. list of the words with high influence scores is below:

|     |       Terms        | Scores |
| :-: | :----------------: | :----: |
|  1  |  $\text{selling}$  | 0.769  |
|  2  |    $\text{yes}$    | 0.665  |
|  3  |   $\text{would}$   | 0.660  |
|  4  |   $\text{port}$    | 0.576  |
|  5  |    $\text{no}$     | 0.571  |
|  6  |   $\text{added}$   | 0.557  |
|  7  |    $\text{low}$    | 0.528  |
|  8  |   $\text{high}$    | 0.500  |
|  9  |    $\text{few}$    | 0.500  |
| 10  | $\text{quantity}$  | 0.500  |
| 11  |  $\text{Compare}$  | 0.499  |
| 12  |  $\text{expire}$   | 0.496  |
| 13  |   $\text{last}$    | 0.490  |
| 14  |  $\text{limited}$  | 0.480  |
| 15  |  $\text{demand}$   | 0.477  |
| 16  |   $\text{risk}$    | 0.467  |
| 17  |   $\text{sell}$    | 0.451  |
| 18  | $\text{purchased}$ | 0.437  |
| 19  |  $\text{already}$  | 0.427  |
| 20  |   $\text{only}$    | 0.419  |
| 21  |  $\text{bought}$   | 0.409  |
| 22  |   $\text{sold}$    | 0.386  |
| 23  |   $\text{less}$    | 0.384  |
| 24  | $\text{withdraw}$  | 0.375  |
| 25  | $\text{remaining}$ | 0.345  |

## License

- [Apache-2.0 license](https://github.com/yamanalab/why-darkpattern?tab=Apache-2.0-1-ov-file#readme)
