# %% [markdown]
# 
# # 1. INTRODUCTION
# 
# > ## 1.1 Problem Definition -
# Lung Cancer is found to be the second most common cancer in the world. Every year many people are diagnosed with lung cancer and most people die from lung cancer due to late diagnosis since it does not show much symptoms in the early stages. Statistics have shown that if lung cancer is found at an earlier stage, when it is small and before it has spread, it is more likely to be treated successfully. Thus, we have created a machine learning model using Logistic Regression that helps in early detection of lung cancer by increasing the chances of survival of patients suffering from it. 
# 
# > ## 1.2 Need -
# 1. Increase in Lung Cancer Cases
# 2. Early Diagnosis
# 3. Decreases Mortality
# 
# 

# %%
#Importing Libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#For ignoring warnings
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ---

# %% [markdown]
# # 2. DATASET
# 
# ##2.1 Data Understanding
# Knowing everything about the dataset is extremely important to ensure the proper analysis of the machine learing model using the attributes of the data.
# 
# - **Collected From :** [Kaggle - Lung Cancer Prediction](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)
# - **Size :** 33.9 kB
# - **Type :** csv file
# - **Columns :** 24
# - **Attributes :** 668
# - **Attributes Details :**  
# > 1. Age : (Numeric)
# > 2. Gender : M(male), F(female) 
# > 3. Air Pollution : (Categorical)
# > 4. Alchohol use : (Categorical)
# > 5. Dust Allergy : (Categorical)
# > 6. Occupational Hazards : (Categorical)
# > 7. Genetic Risk : (Categorical)
# > 8. Chronic Lung Disease : (Categorical)
# > 9. Balanced Diet : (Categorical)
# > 10. Obesity : (Categorical)
# > 11. Smoking : (Categorical)
# > 12. Passive Smoker : (Categorical)
# > 13. Chest Pain : (Categorical)
# > 14. Coughing Blood : (Categorical)
# > 15. Fatigue : (Categorical)
# > 16. Weight loss : (Categorical)
# > 17. Shortness of Breath : (Categorical)
# > 18. Wheezing : (Categorical)
# > 19. Swallowing Difficulty : (Categorical)
# > 20. Clubbing of Finger Nails : (Categorical)
# > 21. Frequent Cold : (Categorical)
# > 22. Dry Cough : (Categorical)
# > 23. Snoring : (Categorical)
# > 24. Lung Cancer : Yes , No
# 
# - **Independent Variables :** Age, Gender, Air Pollution, Alcohol use, Dust Allergy, Occupational Hazards, Genetic Risk, Chronic Lung Disease, Balanced Diet, Obesity, Smoking, Passive Smoker, Chest Pain, Coughing Blood, Fatigue, Weight Loss, Shortness of Breath, Wheezing, Swallowing Difficulty, Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring
# - **Dependent Variable :** Lung Cancer (Yes, No)

# %%
#Loading the Dataset
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

df = pd.read_csv(resource_path("dataset.csv"))
df

# %%
#Prints the first five rows
df.head()

# %%
#Summary of the dataset
df.info()

# %% [markdown]
# ## 2.2 Data Preprocessing
# 
# Data Preprocessing is important to identify and correctly handle the missing values, failing to do this, you might draw inaccurate and faulty conclusions and inferences from the data. Hence, it ensures an increase in the accuracy and efficiency of a machine learning model. This was achieved by checking if the dataset has any missing values, if it contains duplicate values (and if yes then remove it), then checking if it has any null values or not. Lastly, we also check if there is uniformity in the data types of the attributes.
# 

# %%
#Size of the dataset
df.shape

# %%
#Checking for Missing Values
percent = (df.isnull().sum()).sort_values(ascending=False)
percent.plot(kind="bar", figsize = (14,6), fontsize = 10, color='blue')
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Missing Count", fontsize = 20)
plt.title("Total Missing Value ", fontsize = 20)

# %%
#Checking for Duplicates
df.duplicated().sum()

# %%
#Removing Duplicates
df=df.drop_duplicates()

# %%
#Dataset after removing duplicates
df.shape

# %%
#Checking for null values
df.isnull().sum()

# %%
#Description of the dataset (only provides info for numerical columns)
df.describe()

# %%
#Checking the datatype to normalize the labels
df.dtypes

# %%
#Finding unique elements
df["Lung cancer"].unique()

# %%
#Finding unique elements
df["Gender"].unique()

# %% [markdown]
# Since the dataset contains two columns 'Gender' and 'Lung cancer' attributes which are of object data type. So, we convert them to numerical values. 
# 
# Conversion :
# - Gender - Female = 0, Male = 1
# - Lung cancer - NO = 0, YES = 1

# %%
#Mapping numeric values to non-numeric values
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['Lung cancer'] = df['Lung cancer'].map({'No': 0, 'Yes': 1})

# %%
#Checking the datatype after conversion 
df.dtypes

# %%
df.head()

# %%
df.info()

# %% [markdown]
# ---

# %% [markdown]
# #3. EXPLORATORY ANALYSIS
# 
# Exploratory data analysis (EDA) involves using statistics and visualizations to analyze and identify trends in data sets. It is used for seeing what the data can tell us before the modeling task.
# 

# %%
#Checking the distributaion of Target variable
sns.countplot(x='Lung cancer', data=df,)
plt.title('Target Distribution');

# %%
#Count of the Target variable values
df['Lung cancer'].value_counts()

# %%
#Function for plotting
def plot(col, df=df):
    return df.groupby(col)['Lung cancer'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))

# %%
plot('Age')

# %%
plot('Gender')

# %%
plot('Air Pollution')

# %%
plot('Alcohol use')

# %%
plot('Dust Allergy')

# %%
plot('OccuPational Hazards')

# %%
plot('Genetic Risk')

# %%
plot('chronic Lung Disease')

# %%
plot('Balanced Diet')

# %%
plot('Obesity')

# %%
plot('Smoking')

# %%
plot('Passive Smoker')

# %%
plot('Chest Pain')

# %%
plot('Coughing Blood')

# %%
plot('Fatigue')

# %%
plot('Weight Loss')

# %%
plot('Shortness of Breath')

# %%
plot('Wheezing')

# %%
plot('Swallowing Difficulty')

# %%
plot('Clubbing of Finger Nails')

# %%
plot('Frequent Cold')

# %%
plot('Dry Cough')

# %%
plot('Snoring')

# %% [markdown]
# #4. Model Building 

# %%
#Splitting independent and dependent variables
X = df.drop('Lung cancer', axis = 1)
y = df['Lung cancer']

# %%
#Standardizing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)


X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

X_train, X_val, y_train, y_val = train_test_split(
    X_bias, y, test_size=0.2, random_state=42, stratify=y
)

# %% [markdown]
# #Sigmoid Function

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sns.kdeplot(y_val,cumulative=True, bw=1.5)

# %% [markdown]
# #Log Loss Function

# %%
def compute_loss(w, X, y):
    z = X @ w
    # applying the sigmoid function to get predicted probabilities
    predictions = sigmoid(z)
    # binary cross-entropy loss
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return loss

# %% [markdown]
# #Gradient of the Loss Function

# %%
def compute_gradient(w, X, y):
    z = X @ w
    predictions = sigmoid(z)
    # error = difference between predicted and true labels
    errors = predictions - y
    # compute the gradient of the loss function
    gradient = X.T @ errors / len(y)
    return gradient

# %% [markdown]
# #Validation Accuracy Function

# %%
def validation_accuracy(w, X_val, y_val):
    probabilities = sigmoid(X_val @ w)
    predictions = (probabilities > 0.5).astype(int) # decission rule for binary classification
    accuracy = np.mean(predictions == y_val)
    return accuracy

# %% [markdown]
# #Gradient Descent Optimization

# %%
def gradient_descent_logistic(X_train, y_train, X_val, y_val,
                              learning_rate=0.1, n_steps=1000, tolerance=1e-6):
    w = np.zeros(X_train.shape[1])  # start with all weights equal to 0
    loss_history = [compute_loss(w, X_train, y_train)]
    val_accuracy_history = [validation_accuracy(w, X_val, y_val)]
    weights_history = [w.copy()]  # storing weights for decision boundary plotting

    for step in range(1, n_steps + 1):
        grad = compute_gradient(w, X_train, y_train)
        w -= learning_rate * grad  # update rule
        loss = compute_loss(w, X_train, y_train)
        loss_history.append(loss)

        # compute validation accuracy
        acc = validation_accuracy(w, X_val, y_val)
        val_accuracy_history.append(acc)

        # storing weights every 10 steps for plotting
        if step % 10 == 0:
            weights_history.append(w.copy())

        # check convergence
        if np.abs(loss_history[-2] - loss_history[-1]) < tolerance:
            print(f'Converged at step {step}')
            break

        if step % 100 == 0:
            print(f'Step {step}: Loss = {loss:.4f}, Validation Accuracy = {acc:.4f}')

    return w, loss_history, val_accuracy_history, weights_history

# %% [markdown]
# #5. Model Training and Interpretation 

# %%
learning_rate = 0.05
n_steps = 800

w_opt, loss_history, val_accuracy_history, weights_history = gradient_descent_logistic(
    X_train, y_train, X_val, y_val,
    learning_rate=learning_rate,
    n_steps=n_steps,
)


# %% [markdown]
# #6. CONCLUSION
# 
# As Lung Cancer is the most common cancer in the world and statistics have been proved that detection in early stages have greater chances of getting cured. Thus, we have successfully developed a machine learning model using Logistic Regression for early detection of Lung Cancer. Our model helps in early detection of Lung Cancer which will help save the lives of the patients because with early detection and diagnosis of Lung Cancer there are higher chances of survival for the patients.
# 


