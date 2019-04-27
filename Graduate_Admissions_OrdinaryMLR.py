"Graduate Admissions Prediction Using Multiple Linear Regression"
"Midterm Project for Multivariate Statisfied Analysis"
"By : Olivia Ferlita [M10702818]"

"- Ordinary MLR -"

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error

# Importing the dataset
data = pd.read_csv("Admission_Predict_Ver1.1.csv")
data = data.drop(['Serial No.'], axis=1)
x = data.iloc[:, :-1].values
y = data.iloc[:, 7].values

# Making the correlation matrix
correlation = data.corr()
plt.figure(figsize=(13, 11))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="coolwarm", annot_kws={"size":16})
plt.savefig('Correlation_Matrix.png', format='png')

# Histogram of Dependent Variable
'Chance of Admit'
plt.figure(figsize = (6,4))
plt.hist(y, bins = 10, histtype = 'bar', color = 'pink', edgecolor = 'black', linewidth = 1)
plt.xlabel("Chance of Admit", fontsize = 14)
plt.ylabel("Frequency", fontsize = 14)
plt.savefig('Histogram of Chance of Admit.png', format='png')

# Histogram of Independent Variable
'GRE Score'
plt.figure(figsize = (6,4))
plt.hist(data.iloc[:, 0], bins = 10, histtype = 'bar', color = 'steelblue', edgecolor = 'black', linewidth = 1)
plt.xlabel("GRE Score", fontsize = 14)
plt.ylabel("Frequency", fontsize = 14)
plt.savefig('Histogram of GRE score.png', format='png')

'TOEFL Score'
plt.figure(figsize = (6,4))
plt.hist(data.iloc[:, 1], bins = 10, histtype = 'bar', color = 'steelblue', edgecolor = 'black', linewidth = 1)
plt.xlabel("TOEFL Score", fontsize = 14)
plt.ylabel("Frequency", fontsize = 14)
plt.savefig('Histogram of TOEFL score.png', format='png')

'CGPA'
plt.figure(figsize = (6,4))
plt.hist(data.iloc[:, 5], bins = 10, histtype = 'bar', color = 'steelblue', edgecolor = 'black', linewidth = 1)
plt.xlabel("CGPA", fontsize = 14)
plt.ylabel("Frequency", fontsize = 14)
plt.savefig('Histogram of CGPA.png', format='png')

# Plot of the highly correlated features toward Dependent Variable
'GRE Score'
sns.set(font_scale=1.5)
plot = sns.jointplot(data.iloc[:, 0], data.iloc[:, 7], kind='reg',joint_kws={'line_kws':{'color':'red'}}, height=10)
plot.savefig('GRE Score vs Chance of Admit.png', format='png')

'TOEFL Score'
sns.set(font_scale=1.5)
plot = sns.jointplot(data.iloc[:, 1], data.iloc[:, 7], kind='reg',joint_kws={'line_kws':{'color':'red'}}, height=10)
plot.savefig('TOEFL Score vs Chance of Admit.png', format='png')

'CGPA'
sns.set(font_scale=1.5)
plot = sns.jointplot(data.iloc[:, 5], data.iloc[:, 7], kind='reg',joint_kws={'line_kws':{'color':'red'}}, height=10)
plot.savefig('CGPA vs Chance of Admit.png', format='png')

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Plot the Predicted Value
plt.figure(figsize = (15,5))
plt.grid(linestyle='--')
plt.plot(range(len(y_test)),y_test, color = 'b', label = 'Real Value')
plt.plot(range(len(y_pred)), y_pred, color = 'orange', label = 'Predicted Value')
plt.xlabel("Sample", fontsize=14)
plt.ylabel("Chance of Admit", fontsize=14)
plt.legend()
plt.savefig('y_pred vs y_test.png', format='png')

# Calculating Residual Error
eHat = y_pred - y_test

# Plotting Residual Error
plt.figure(figsize = (10,5))
plt.grid(linestyle='--')
b, m = polyfit(y_pred,eHat, 1)
plt.scatter(y_pred,eHat)
plt.plot(y_pred, b + m * y_pred, '--', color='r')
plt.xlabel("y_pred", fontsize=14)
plt.ylabel("eHat", fontsize=14)
plt.savefig('eHat vs y_pred.png', format='png')

# Model Evaluation
def evaluation(ytest,ypred):
    'Adjusted R-squared'
    R2 = r2_score(ytest,ypred)
    adjR2 = 1-(1-R2)*(len(x_test)-1)/(len(x_test)-x_test.shape[1]-1)
    'F Test'
    nf = sum((ypred-np.mean(ytest))**2)/x_test.shape[1]
    df = sum((ytest-ypred)**2)/(len(x_test)-x_test.shape[1]-1)
    F = nf/df
    'MAE'
    MAE = mean_absolute_error(ytest, ypred)
    return adjR2, F, MAE
score = np.zeros((1,3))
score[0,:] = np.array([evaluation(y_test, y_pred)])
#
#
#
#
#