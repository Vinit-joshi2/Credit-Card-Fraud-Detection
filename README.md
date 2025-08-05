# Credit-Card-Fraud-Detection

<h4>
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
</h4>

<p>
  The<a href= "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">DataSet</a> contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation.
More background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
</p>


## Our Goal

<p>
- Understand the  distribution of the  data that was provided to us.
  
- Handle Imbalanced Data

- Create a 50/50 sub-dataframe ratio of "Fraud" and "Non-Fraud" transactions

- Determine the Classifiers Model we are going to use and decide which one has a higher accuracy.
  
</p>

## Requirements

- **Python Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` , `sklearn`
- **Kaggle API Key** (for data downloading)

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
2. Install Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Kaggle API, download the data  and follow the steps to load
   
    - Run this command on any code editor

      
   ```
     !pip install opendatasets
   ```

   
    - Once you install library then run below command  for loading data into your editor for that you need to pass your kaggle credentials
      
   ```
       import opendatasets as od
       url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data"
       od.download(url)
   ```

## Understand the Data

<p>
  The transaction amount is relatively small. The mean of all the amounts made is approximately USD 88. There are no `Null` values, so we don't have to work on ways to replace values. Most of the transactions were Non-Fraud (99.83%) of the time, while Fraud transactions occurs (017%)of the time in the dataframe.
</p>


```
print("No Frauds" , df["Class"].value_counts()[0] / df.shape[0] *100 , "% of the datasets")
print("Frauds" , df["Class"].value_counts()[1] / df.shape[0] *100 , "% of the dataset")

```
<img src = "img1">

<img src = "img2">

<p>
  How imbalanced is our original dataset! Most of the transactions are non-fraud. If we use this dataframe as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will "assume" that most transactions are not fraud. But we don't want our model to assume, we want our model to detect patterns that give signs of fraud!
</p>



## Random Under-Sampling

<img src = "img3">

<p>
- The first thing we have to do is determine how imbalanced is our class (use value_counts() on the class column to determine the amount for each label)

- Once we determine how many  are considered fraud transactions (Fraud = "1") , we should bring the non-fraud transactions to the same amount as fraud transactions (assuming we want a `50/50` ratio), this will be equivalent to `492 cases of fraud` and `492 cases of non-fraud` transactions.
  
- After implementing this technique, we have a sub-sample of our dataframe with a 50/50 ratio with regards to our classes. Then the next step we will implement is to shuffle the data to see if our models can maintain a certain accuracy everytime
  
</p>

```
''' we know that our data is highly skewed we should make our data eqivalent(50/50 ratio data) to
    have a normal distrubution of th classes
'''

df = df.sample(frac = 1)

fraud_df = df.loc[df["Class"] ==1]
non_fraud_df = df.loc[df["Class"] ==0][:492]

normal_distributed_df = pd.concat([fraud_df , non_fraud_df])

new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()

```

```
new_df["Class"].value_counts().reset_index()

```

<img src = "img4">

<p>
  Now we created sub-sample which is equally distributed and contain equal number of data from each classes
</p>

<img src = "img5">

<p>

Correlation Matrices - 

Correlation matrices are the essence of understanding our data. We want to know if there are features that influence heavily in whether a specific transaction is a fraud. However, it is important that we use the correct dataframe (subsample) in order for us to see which features have a high positive or negative correlation with regards to fraud transactions.
</p>

<img src = "img6">

<p>

Summary and Explanation:

- Negative Correlations: V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.
  
- Positive Correlations: V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.
  
- BoxPlots: We will use boxplots to have a better understanding of the distribution of these features in fradulent and non fradulent transactions.
  
</p>

<img src = "img7">

<img src = "img8">

<img src = "img9">


<p>
  
  Visualize Distributions: We first start by visualizing the distribution of the feature we are going to use to eliminate some of the outliers. V14 is the only feature that has a Gaussian distribution compared to features V12 and V10.
  
</p>

## Split the Data
<p>

  we will train four types of classifiers and decide which classifier will be more effective in detecting fraud transactions. Before we have to split our data into training and testing sets and separate the features from the labels.


</p>
