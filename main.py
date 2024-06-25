import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,mean_squared_error,confusion_matrix

df = pd.read_csv('loan_data_set.csv')

df.shape
df.describe()
df.info()

# show categorical data in the dataset and its values
def unique_data():
 print(f'the unique values of Dependents: {df["Dependents"].unique()}')
 print(f'the unique values of Loan_Amount_Term: {sorted(df["Loan_Amount_Term"].unique())}')
 print(f'the unique values of Property_Area: {df["Property_Area"].unique()}')
 print(f'the unique values of Gender: {df["Gender"].unique()}')
 print(f'the unique values of Married: {df["Married"].unique()}')
 print(f'the unique values of Education: {df["Education"].unique()}')
 print(f'the unique values of Self_Employed: {df["Self_Employed"].unique()}')
 print(f'the unique values of Credit_History: {df["Credit_History"].unique()}')
unique_data()

sum(df.duplicated(subset = 'Loan_ID')) == 0
df.drop("Loan_ID",axis=1,inplace=True)
# check the missing values
df.isnull().sum()

missing =['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term','Credit_History']
for i in missing[:7]:
  # fill the categorical data with the mode of the column
    df[i].fillna(df[i].mode()[0],inplace=True)

#fill the continuous data with the mean to avoid outliers
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

df.isnull().sum()

# check outliers
def print_ol():
 sns.boxplot(data=df , x = 'ApplicantIncome' , hue = 'Loan_Status')
 plt.show()
 sns.boxplot(data=df , x = 'CoapplicantIncome' , hue = 'Loan_Status')
 plt.show()
 sns.boxplot(data=df , x = 'LoanAmount' , hue = 'Loan_Status')
 plt.show()
print_ol()

columns = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
for col in columns:
    # calculate interquartile range
    q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    iqr = q75 - q25
    lower= q25-(iqr * 1.5)
    upper= (iqr*1.5)+q75
    # identify outliers
    outliers=( ( df[col] < lower) | (df[col] > upper) )
    outlier_index= df[outliers].index
    df.drop(outlier_index,inplace=True)

print_ol()

#data visualization

col=['Gender','Married','Dependents','Education','Self_Employed','Credit_History']
for i in col[:7]:
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Loan_Status', data=df ,palette='plasma')
    plt.xlabel(i, fontsize=14)
    
y = df['Property_Area']
d=df['Property_Area'].value_counts()
#create a dictionary
d=dict(d)
mylabels = ['U', 'R', 'S']
plt.pie(d.values(), labels = mylabels,autopct='%1.1f%%')
plt.legend(title="category")

y = df['Dependents']
d=df['Dependents'].value_counts()
d=dict(d)
mylabels = ['0', '1','2','+3']
plt.pie(d.values(), labels = mylabels,autopct='%1.1f%%')
plt.legend(title="Dependents")

y = df['Credit_History']
d=df['Credit_History'].value_counts()
d=dict(d)
mylabels = ['0', '1']
plt.pie(d.values(), labels = mylabels,autopct='%1.1f%%')
plt.legend(title="Credit History")

y = df['Education']
d=df['Education'].value_counts()
d=dict(d)
mylabels = ['Graduate', 'Not Graduate']
plt.pie(d.values(), labels = mylabels,autopct='%1.1f%%')
plt.legend(title="category")

y = df['Married']
d=df['Married'].value_counts()
d=dict(d)
mylabels = ['Married', 'Not Married']
plt.pie(d.values(), labels = mylabels,autopct='%1.1f%%')
plt.legend(title="category")

y = df['Gender']
d=df['Gender'].value_counts()
d=dict(d)
mylabels = ['Male', 'Female']
plt.pie(d.values(), labels = mylabels,autopct='%1.1f%%')
plt.legend(title="category")

col2=['Gender','Married','Dependents','Education','Self_Employed','Loan_Amount_Term','Credit_History','Property Area']
# display count figure for each column except the last one
for i in col2[:-1]:
  # figsize = ( width , length ) in pixels
 plt.figure(figsize = (5,3))
 sns.countplot(  data = df , x = i )
plt.show()


sns.displot(df['Loan_Amount_Term'])
sns.pairplot(df)


#encoding categorical data

cols_to_encode = ['Education', 'Married','Gender','Self_Employed','Loan_Status','Loan_Amount_Term','Property_Area']

label_encoder = LabelEncoder()
for col in cols_to_encode:
    df[col] = label_encoder.fit_transform(df[col])
    
df['Dependents'] = df['Dependents'].replace(['3+'], [int('3')])
df['Dependents'] = df['Dependents'].replace(['0'], [int('0')])
df['Dependents'] = df['Dependents'].replace(['1'], [int('1')])
df['Dependents'] = df['Dependents'].replace(['2'], [int('2')])

#Feature Selection and Extraction

# using correlation
df.corr()
sns.set(rc={'figure.figsize': (15, 8)})
sns.heatmap(df.corr(),annot=True)

#using chi-squared
dfs=df[['Gender', 'Married', 'Dependents','Education', 'Self_Employed', 'Loan_Amount_Term',  'Credit_History',  'Property_Area', 'Loan_Status']]
x_temp=dfs.drop("Loan_Status",axis=1)
y_temp=dfs['Loan_Status']
chi_scores=chi2(x_temp,y_temp)
chi_scores

#the higher the chi value, the more important the feature
chi_values=pd.Series(chi_scores[0],index=x_temp.columns)
chi_values.sort_values(ascending=False,inplace=True)
chi_values.plot.bar()

#the lower the p value the more important the feature
p_values=pd.Series(chi_scores[1],index=x_temp.columns)
p_values.sort_values(ascending=False,inplace=True)
p_values.plot.bar()

df.drop("Loan_Amount_Term",axis=1,inplace=True)
df.drop("Dependents",axis=1,inplace=True)
df.drop("Self_Employed",axis=1,inplace=True)



X = df.iloc[:, 0:8].values
Y = df.iloc[:, 8].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state=0)

income = df['ApplicantIncome']
coincome = df['CoapplicantIncome']
lamount=df['LoanAmount']

#Normalization
scaler = MinMaxScaler()
income_scaled = scaler.fit_transform(income.values.reshape(-1,1))
lamount_scaled = scaler.fit_transform(lamount.values.reshape(-1,1))

#Standardization
scaler=StandardScaler()
coincome_scaled=scaler.fit_transform(coincome.values.reshape(-1,1))
df['ApplicantIncome'] = income_scaled
df['LoanAmount']=lamount_scaled
df['CoapplicantIncome']=coincome_scaled

df.head(10)




#Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# using SVM model
clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

# using Decision Tree model
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

# using XG boost model
model = XGBClassifier(max_depth=3, subsample=1, n_estimators=50, min_child_weight=1, random_state=5)
model.fit(X_train, y_train)




#Testing

# Using Logistic Regression Model
y_pred = classifier.predict(X_test)
logisticAcc=accuracy_score(y_pred,y_test)*100
print('Classification report: \n',classification_report(y_test, y_pred))
print(f"Accuracy: {round(logisticAcc,2)}%")
print('Mean Squared Error : ', round(mean_squared_error(np.asarray(y_test), y_pred),2))
print('Confusion matrix :\n ', confusion_matrix(y_test,y_pred))


# Using SVM Model
clf = SVC(kernel='linear',C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
svm_acc = accuracy_score(y_pred,y_test)*100
print('Classification report: \n',classification_report(y_test, y_pred))
print(f"Accuracy: {round(svm_acc,2)}%")
print('Mean Squared Error : ', round(mean_squared_error( np.asarray(y_test), y_pred),2))
print('Confusion matrix : \n', confusion_matrix(y_test,y_pred))

# Using Decision Tree Model
y_predict = DT.predict(X_test)
decisionAcc = accuracy_score(y_predict,y_test)*100
print('Classification report: \n',classification_report(y_test, y_predict))
print(f"Accuracy: {round(decisionAcc,2)}%")
print('Mean Squared Error : ', round(mean_squared_error(np.asarray(y_test), y_predict),2))
print('Confusion matrix : \n', confusion_matrix(y_test,y_pred))


# Using XG Boost Model
y_predict = model.predict(X_test)
xgbAcc=accuracy_score(y_test, y_predict)*100
print('Classification report:\n',classification_report(y_test, y_predict))
print(f"Accuracy: {round(xgbAcc,2)}%")
print('Mean Squared Error : ',round( mean_squared_error(np.asarray(y_test), y_predict),2))
print('Confusion matrix :\n', confusion_matrix(y_test,y_pred))



score = [ round(xgbAcc,2), round(logisticAcc,2) , round(svm_acc,2) ,round(decisionAcc,2)]
Models = pd.DataFrame({'Algorithm': ["XG boost","Logistic Regression","SVM","Decision Tree"],'Accuracy': score})
Models.sort_values(by='Accuracy', ascending=False)