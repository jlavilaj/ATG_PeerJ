# Academic Trajectory Graph (ATG)

Official implementation accompanying the paper:

**Predicting Academic Success with Graph Neural Networks**

------------------------------------------------------------------------

## Overview

This repository provides the reference implementation of the **Academic
Trajectory Graph (ATG)**, a graph-based representation designed to model
student academic progression across semesters.

The framework predicts student success using Graph Neural Networks and
compares their behavior against conventional machine learning
baselines.\
The goal is not only classification performance, but also the analysis
of academic pathways and progression patterns.

------------------------------------------------------------------------

## Implemented Components

The repository includes:

-   Construction of the Academic Trajectory Graph (ATG)
-   Graph Neural Network models:
    -   Graph Convolutional Network (GCN)
    -   Deep Graph Convolutional Neural Network (DGCNN)
-   Random Walk / node2vec graph embeddings
-   Tabular baselines:
    -   Logistic Regression
    -   Linear Support Vector Machine
    -   Random Forest
    -   Multilayer Perceptron (MLP)
-   Statistical evaluation:
    -   Friedman test
    -   Nemenyi post-hoc comparison

------------------------------------------------------------------------

## Dataset

The dataset used in the paper consists of institutional academic records
from a university information system.

Due to privacy and data protection regulations, the dataset **cannot be
publicly released**.

However, the full experimental pipeline can be reproduced on any dataset
that contains the following minimal structure:

    student_id, semester, course_id, grade, credits

Each record should represent a course taken by a student in a specific
semester.

------------------------------------------------------------------------

## Evaluation Protocol

All experiments in the paper were conducted using:

-   **Student-level 10-fold cross-validation**
-   Graph reconstruction inside each fold
-   Fold-specific preprocessing and normalization
-   Independent model initialization per fold

No information from test students is used during training.

This protocol prevents data leakage and provides an unbiased estimate of
generalization performance.

------------------------------------------------------------------------

## Installation

Create a virtual environment (recommended) and install the dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Running the Code

The main entry point describing the experimental workflow:

    python run_experiment.py

The script outlines the steps followed in the experiments: 1.
Cross-validation splitting by student 2. ATG construction 3. Model
training 4. Metric computation 5. Statistical comparison

------------------------------------------------------------------------

## Ethical Use

The models are intended as **decision-support tools** for academic
advising and early intervention.\
They are **not** designed for automated student evaluation or
administrative decision-making.

Predictions must always be interpreted alongside human academic
guidance.

------------------------------------------------------------------------

## Citation

If you use this repository in your research, please cite the associated
paper:

    @article{ATG2026,
    ...
    }

------------------------------------------------------------------------

## License

This project is released under the MIT License.
