# 1-Prediction-using-Supervised-ML
# Importing all libraries required in this notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# Read data 
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data=pd.read_csv(url)
data.head(25)
#data.shape
# Plotting the distribution of scores
data.plot(x="Hours",y="Scores",style='o')
plt.title("Hours vs Percentages")
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()
# Preparing the data The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
print(X)
print(y)
# the next step is to split this data into training and test sets.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.1, random_state=1) 
print(X_train)
print(X_test)
# now is finally the time to train our algorithm.
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='y');
plt.show()
# it's time to make some predictions.
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test)
print(y_pred)
print(y_test)
# Compare the actual and predicted percentages.
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df # for print table
# You can also test with your own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours studied",hours,"precentage is",own_pred)
# The final step is to evaluate the performance of algorithm.
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
# complete      
