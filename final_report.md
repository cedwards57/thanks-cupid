# Thanks, Cupid: Final Report

Georgia State Univeristy

CSC 4780 - Fundamentals of Data Science, Fall 2022

Presented by the Cupid Scientists:
- Lorena Burrell
- Capella Edwards
- Laurel Sparks
- Venkata Mani Mohana Rishitha Srikakulapu
- Cindy Thai

## Table of Contents
1. [Business Understanding](#1-business-understanding)
- 1.1 [Business Problem](#11-business-problem)
- 1.2 [Dataset](#12-dataset)
- 1.3 [Proposed Analytics Solution](#13-proposed-analytics-solution)
2. [Data Exploration and Preproceesing](#2-data-exploration-and-preproceesing)
- 2.1 [Data Quality Report](#21-data-quality-report)
- 2.2 [Missing Values and Outliers](#22-missing-values-and-outliers)
- 2.3 [Normalization](#23-normalization)
- 2.4 [Feature Selection and Transformation](#24-feature-selection-and-transformation)
3. [Model Selection and Evaluation](#3-model-selection-and-evaluation)
- 3.1 [Evaluation Metrics](#31-evaluation-metrics)
- 3.2 [Models](#32-models)
- 3.3 [Evaluation](#33-evaluation)
    - 3.3.1 [Evaluation Settings and Sampling](#331-evaluation-settings-and-sampling)
    - 3.3.2 [Hyper-parameter Optimization](#332-hyper-parameter-optimization)
    - 3.3.3 [Final Evaluation](#333-final-evaluation)
4. [Results and Conclusion](#4-results-and-conclusion)


## 1 Business Understanding
### 1.1 Business Problem
The purpose of dating applications is to help users find potential romantic partners to connect with. Users are connected based on the nearest location with limited filtering functionality to return a portion of neighboring users. Our app will provide premium services to allow the user to generate a more refined list of matches that have been filtered based on the preferences of both users. To sell these premium services, our model must provide useful results, meaning dates that are likely to retain interest in one another and stay together. Therefore, our app will benefit from predicting which people are most likely to match with each other. It can also stand to profit from selecting which information about its users is most critical to keep to be successful, saving data space and retaining user attention by avoiding a long ‘interview quiz’ portion. This will yield more profits for the business to sell more premium services, and a high success rate will draw more users to join the app which will increase the variety of dating partners for paying customers.

### 1.2 Dataset
The dataset that we selected recorded the results of a speed dating experiment conducted by Columbia Business School by research professors Raymond Fisman, Sheena S. Iyengar, Emir Kamenica, and Itamar Simonson during May 2006. We selected this particular dataset because it has a wide range of descriptive features with continuous and categorical aspects with over eight thousand instances.  This dataset will be useful in making predictions and training our model since we have plenty of data to use as a training and testing set and allow our machine learning algorithm to predict matches between participants. The original speed dating dataset has approximately one-hundred ninety-five features and eight thousand three-hundred and seventy-nine instances. We decided to narrow down the features to a list of nineteen, many of which are derived combinations of the original raw features, for the most impactful and relevant data for selecting matches.

The features we decided to select include the target variable of whether a match was made, expected satisfaction with dating, whether both participants match race/ethnicity, average importance of matching race to both participants, and whether both participants stated the same desired outcome from dating. We also derived the difference between the participants’ interest in physical hobbies, other outdoor hobbies, and indoor hobbies, along with the difference between the participants’ self-rating vs. the other’s desire for qualities of ambition, attractiveness, sincerity, intelligence, funniness, and ambition. Difference in income and age, along with whether both have a desired career in the same area, were also included. We marked the percentage of prospective dates a person expected to be interested in them as “confidence.” The last two features indicate a ranking of how often the participants go on dates and go out in general. Based on the sample scorecard and synopsis of the data, we determined that these features were the most influential in drawing conclusions on matches. We believe that these features will help us narrow down our target feature and train our model to make better predictions on matches.

### 1.3 Proposed Analytics Solution
With our subset dataset, we plan to use a supervised similarity-based machine learning algorithm to predict successful matches. Our first step in accomplishing this task was to create a subset of key descriptive features from our larger dataset. We choose nineteen descriptive features, including features like happiness expectation, hobby/interest value, goal difference, and more (see table in the appendix for the complete table of key descriptive features and descriptions). Our target variable is our match feature which predicts the outcome as 0 for no match and 1 for a match between two individuals participating in speed-dating.

## 2 Data Exploration and Preproceesing
### 2.1 Data Quality Report
Below, we have listed the count, cardinality, and percentage of missing values for all features. For categorical features, we calculated the first and second mode, mode frequency, and mode percentage. For continuous features, we calculated the minimum, maximum, quartiles, and standard deviation.


### 2.2 Missing Values and Outliers
### 2.3 Normalization
### 2.4 Feature Selection and Transformation
## 3 Model Selection and Evaluation
### 3.1 Evaluation Metrics
### 3.2 Models
### 3.3 Evaluation
#### 3.3.1 Evaluation Settings and Sampling
#### 3.3.2 Hyper-parameter Optimization
#### 3.3.3 Final Evaluation
## 4 Results and Conclusion