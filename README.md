# Thanks, Cupid

A data science project for a hypothetical dating app.

## Dataset

We explore the [dataset](http://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/) gathered from a speed dating experiment, for the [Gender Differences in Mate Selection](https://academic.oup.com/qje/article/121/2/673/1884033) experiment conducted from 2002-2004 by Columbia Business School professors, Fisman et. al.

Although this dataset was originally used to analyze gender differences, we found it useful for predicting matches between pairs of people, regardless of potential gendered behaviors.

## Project

In this project, we used the dataset to build a machine learning model for a hypothetical dating app, which would optimize predictions based on the interest of selling a useful, marketable service.

We reduced the dataset, originally 195 features, to 19 features, many of which were imputed. After handling missing values and normalizing the new data, we experimented with three different types of models. We optimized each model's parameters, then evaluated and compared their final performance for our purposes.

## File Directory

- `data`: folder containing our dataset files, including the original from the experiment, and the reduced and transformed version that arises after running `data_processing.ipynb`.
    - The `data/dataset_adjusted.csv` file is a checkpoint file included for ease of use, as mentioned in `data_processing.ipynb`.
    - `features.csv` is a guide to the reduced set of features.
- `figs`: contains the graphs output by both Jupyter notebook files.
- `cupid_scientists_final_report.pdf`: a final report which thoroughly discusses our initial problem, dataset, methods, and outcomes. 
- `cupid_scientists_presentation.pdf`: the PowerPoint file used for our final presenation.
- `data_processing.ipynb`: a Jupyter notebook covering transformations, processing, and feature reduction with our dataset.
- `modeling.ipynb`: a Jupyter notebook demonstrating our experiments with each machine learning model.
