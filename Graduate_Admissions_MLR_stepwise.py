"Graduate Admissions Prediction Using Multiple Linear Regression"
"Midterm Project for Multivariate Statisfied Analysis"
"By : Olivia Ferlita [M10702818]"

"- Stepwise Method -"

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

# Applying Stepwise method (run code step by step)
import statsmodels.formula.api as sm
'1'
x_opt = x[:,[0,1,2,3,4,5,6]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
'2'
x_opt = x[:,[0,1,2,4,5,6]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 0.2, random_state = 0)

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
plt.savefig('y_pred vs y_test (stepwise).png', format='png')

# Calculating Residual Error
eHat=y_pred-y_test

# Plotting Residual Error
plt.figure(figsize = (10,5))
plt.grid(linestyle='--')
b, m = polyfit(y_pred,eHat, 1)
plt.scatter(y_pred,eHat)
plt.plot(y_pred, b + m * y_pred, '--', color='r')
plt.xlabel("y_pred", fontsize=14)
plt.ylabel("eHat", fontsize=14)
plt.savefig('eHat vs y_pred (stepwise).png', format='png')

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