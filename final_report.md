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
- 2.4 [Transformations](#24-transformations)
- 2.4 [Feature Selection](#25-feature-selection)
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

All of our categorical features are binary, and most show a heavy preference for one or the other. For this reason, we may end up employing under- or oversampling when training our model, particularly for the target feature. For continuous features, most display a healthily skewed unimodal distribution. Income difference and confidence are both exponential, and with how great the standard deviation of the income values are, it is important to handle outliers well.

### 2.2 Missing Values and Outliers
Before we begin the modeling process, we must clean the data by dealing with missing values and outliers.

First, we used imputation to fill in missing values in each continuous column in the dataset. None of our categorical features had any missing values, thankfully. Mean imputation was used for all features except income_diff, which had such a high number of missing values that replacing them all equally meant destroying the original distribution. Instead, we used KNN imputation to preserve a more accurate distribution.

Next, we used Tukey’s Range Test or the Interquartile-Range method to create a function to clamp all outliers in the dataset. We chose not to drop all columns containing outliers, as it would remove around half the rows in the dataset. We also chose not to clamp certain features that were scales guaranteed to be in the range 1-7 or 1-10, such as imprace, out_freq, date_freq, and exphappy. We also did not clamp the confidence feature, as the high number of imputed values made the regular lower and upper bounds have microscopic difference. The same issue happened with income, but due to the high outliers, we chose to clamp this feature with percentiles 0.5 to 0.95 instead of the IQR method.

### 2.3 Normalization
We wanted to avoid bias and inequality between features, especially those that sum up differences between different numbers of features, such as the hobby groups, or those with particularly large values, such as income_diff. For this reason, we used range normalization to give equal footing to all continuous features. Notably, this did not affect the confidence rating, as that was already a percentage rating with some values already at 0 and 1.

### 2.4 Transformations
In the beginning, we had 195 features in our original dataset which were transformed using aggregations and derivations to create a dataset with lesser complexity. Now, we’re using that data to identify how well the prediction of finding a match is done. 

We collected metrics about the distribution of our data so that we can better understand common traits of our users and determine the importance of matching a potential couple that share these traits to improve the efficiency of our prediction. First we separate out of continuous variables and categorical variables, so that we can perform binning on the subsets. We then performed our normalization method MinMaxScaling to which has the benefit of preserving the original relationship between our original distribution. We will then generate a histogram to visualize the distributions of our data for further analysis.

After we normalize the data we determine that intel_diff has a right-skewed unimodal distribution, in which we observe that there is a majority of potential matches with a higher correlation (lower difference) between one’s self-rated intelligence and the other’s desire for intelligence in a partner. The descriptive feature age_diff remained roughly the same in an exponential fashion, which indicates a smaller age gap between potential matches.

The goal of our transformations was to improve the distribution of the continuous variables and create a more uniformly distributed model. Through removing values that were rare and considered outliers, and normalizing the data we were able to achieve these goals and make some preliminary observations.

### 2.5 Feature Selection
To begin,  we performed two feature selection methods: impurity-based univariate feature selection (IUFS), utilizing entropy, gini index, information gain, and information gain ratio. and recursive feature elimination (RFE) using a logistic regression model. We pursued IUFS first since every feature in our dataset is considered separately to one another and comparing the results before and after could give us insight to which features would impact our models’ performance the greatest and which features were not crucial to our evaluation.

Based on the results from calculating our impurity, gain metrics and RFE, we decided that the features that would have the least amount of impact on our model’s performance would be ‘same_goal’ and ‘amb_diff’. ‘Same_goal’ refers to both participants having the same goal in entering the date, and ‘amb_diff’ refers to whether both participants rated themselves as similarly ambitious people.

Once IUFS had been conducted, we removed the ‘same_goal’ feature from our dataset which left us with seventeen descriptive features. Afterwards, we performed recursive feature elimination which utilizes logistic regression to rank our remaining seventeen features based on their KNN accuracy score and we decided to remove the ‘amb_diff’ feature as well to select the descriptive features we will use for model evaluation. As a result, we found that sixteen descriptive features had the highest accuracy scores for KNN classifiers. After using KNN to evaluate accuracy, we selected these sixteen features and discarded the remaining features to perform model selection and evaluation.

Due to our KNN accuracy score improving when the selected features were removed, we concluded that our IUFS and RFE had been successful, and we proceeded to model selection and evaluation.

## 3 Model Selection and Evaluation
### 3.1 Evaluation Metrics
The evaluation metrics we included in our modeling were accuracy, precision, recall, F1 score, Hanssen-Kuipers Skill Score (TSS), and Gilbert Skill Score (GSS). Accuracy score represents the percentage of correct predictions, meaning what percentage of the speed dates stayed together afterwards. Precision score represents the percentage of predicted matches that ended up correct. Recall score represents the percent of actual matches that were correctly predicted.  F1 is the harmonic mean of precision and recall, which measures our model’s overall predictive performance. TSS measures how well the model separates matches from non-matches. Finally, GSS measures how well our predicted matches correspond to true matches accounting for random, lucky predictions.

### 3.2 Models
The models we chose were probability-based Naive Bayes, information-based Decision Tree, and similarity-based K Nearest Neighbors classifiers to predict our data. First, we created training and testing splits based on our sixteen remaining descriptive features found in our feature selection, along with our target variable, ‘match’.  Then we conducted hyperparameter optimization using grid search with accuracy score based on the given criterion, i.e. gini, entropy, or log loss using the training and testing splits we created.

### 3.3 Evaluation
#### 3.3.1 Evaluation Settings and Sampling
Before beginning any modeling, we created our training and testing dataset by splitting off 25% of the data to hold out of training, in order to test our model on that data afterwards.

#### 3.3.2 Hyper-parameter Optimization
For our hyper-parameter optimization, we evaluated each model separately using a grid search. We based this search on accuracy, recording the scores of each model at every possible combination of the selected parameters and deciding on the best arguments to apply at the end.

For optimizing our Decision Tree Model, we searched maximum depth and purity criteria (gini, entropy, or log loss). We found that it scored best at 0.85 with depth 10 using the gini index.

For optimizing our KNN model, we searched the weights preset, leaf sizes, and the Minkowski metric power. We found that the maximum score our KNN model achieved was 0.87 with distance-based weights, leaf size of ten, and Manhattan distance with a power parameter of 1.

For optimizing our Naive Bayes model, we searched for the smoothing variance value. Our optimization found that the Naive Bayes model performed best with 0.84 with a smoothing variance of 0.01, meaning that portion of the largest variance was added to the variance of all features.

In addition, we optimized the threshold parameter for prediction. Applying the parameters found above, we ran each model at every threshold from 0 to 1 with an increment of 0.01, evaluating each based on their GSS, TSS, and F1 scores. We chose an average of these three metrics as our final thresholds. We chose not to include precision and recall in this average because they are each naturally biased towards very low and very high thresholds by definition. We also chose not to include accuracy because our dataset heavily skews towards negative matches, meaning the threshold for optimum accuracy biases towards high thresholds without actually giving us overall better results.


#### 3.3.3 Final Evaluation
Of the three models we tested, the model that performs best is the K Nearest Neighbors classifier. With all three models optimized, it outperformed both the Decision Tree classifier and the Naive Bayes classifier in all metrics we evaluated. The classifier gave us an accuracy of approximately 0.85. Our optimal threshold for KNN is around 0.35.

## 4 Results and Conclusion
In conclusion, we recommend that our dating application utilizes the K Nearest Neighbors model with a decision threshold of 0.35, weights based on distance, a leaf size of 10, and a power parameter of 1. We found that our KNN model was not only the most accurate at 0.85, it also performed better in all other chosen metrics.

In addition, our model’s optimal prediction threshold is low, meaning it is biased towards positive predictions. This is beneficial to our product, as we prefer to draw in customers with the potential of a match than have a lot of users not receive a match at all, discouraging them from using our application. In addition, given multiple potential matches, we can prioritize those with the highest probability scores. Based on the accuracy and overall performance of our KNN model, our dating application can sell these matching high-probability date services behind a premium subscription.