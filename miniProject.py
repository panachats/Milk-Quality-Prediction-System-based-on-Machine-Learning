import numpy as np
import pandas as pd
import seaborn as sns
import time
import io
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


# Read data from csv
data = pd.read_csv(r"D:\walailak\Year_2_Semester_3\pythonProject\mini_project\milknew.csv")
print(data.head()) # แสดงผลข้อมูล 5 แถวแรกจาก DataFrame
# print(data.tail())

# Data Structure
print(data.shape)
print(data.info())
print(data.describe()) # สรุปสถิติพื้นฐานของข้อมูลที่อยู่ใน DataFrame
print(data.isnull().sum()) # เช็คค่า null ใน column
print(data['Grade'].unique())


# Data Preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
label_encoder = LabelEncoder()
data['Grade'] = label_encoder.fit_transform(data['Grade'])
print(data['Grade'].unique())


sns.heatmap(data[["pH", "Temprature", "Taste", "Odor", "Fat ", "Turbidity","Colour"]].corr(), annot=True, fmt='.1g', cmap='coolwarm')
plt.title('Correlation between Academic Percentages')
plt.show()



# outlier detection


# sns.boxplot(x=data2['pH'])
# plt.title("pH before outlier elimination")
# plt.show()

# # Calculatiing the 1st Quartile and the 3rd Quartile of the 'pH' column
# Q1 = data['pH'].quantile(0.25)
# Q3 = data['pH'].quantile(0.75)
# print("The 1st and 2nd quartile value is {0:1f} and {1:1f} respectively".format(Q1,Q3))
#
# # # Calculating the inter-quartile range
# IOR = Q3 - Q1
# # ระยะห่างระหว่างค่า median กับค่า quartile ที่ 25% และ 75%
# print("The value of inter Quartile Range is:",IOR)
#
# # # Finding the Lower Fence and the Upper Fence
# Lower_fence = Q1 - (1.5 * IOR) #  ค่าที่ต่ำที่สุดของข้อมูลที่ยังไม่ถือว่าเป็น outlier
# Upper_fence = Q3 + (1.5 * IOR) #  ค่าที่สูงที่สุดของข้อมูลที่ยังไม่ถือว่าเป็น outlier
# print("Lower Fence value: ",Lower_fence,"\nThe Upper Fence value: ",Upper_fence)
#
# # # Checking the data which have the 'pH' less than the Lower Fence
# # # or greater than the Upper Fence. Basically we are retrieving the outlier here
# # เพื่อหาข้อมูลที่เป็น outlier ของ feature 'pH' เพื่อดูว่าข้อมูลที่มีค่า 'pH' น้อยกว่า Lower_fence หรือมากกว่า Upper_fence มีอะไรบ้าง
# Outliers = data[(data['pH'] < Lower_fence) | (data['pH'] > Upper_fence)]
# print('outlier',Outliers)
#
# # # Checking the data which have the 'pH' within the Lower fence
# # # and Upper Fence Here we are negating the outliler data and printing only
# # the potential data which is within the Lower and Upper Fence
# data2 = data[~((data['pH'] < Lower_fence) | (data['pH'] > Upper_fence))]
# print('data2',data2)

# sns.boxplot(x=data2['pH'])
# plt.title("pH after outlier elimination")
# plt.show()

print('data',data)
from sklearn.model_selection import train_test_split

X = data.drop(columns = ['Grade'])
y = data['Grade']
# print(x.describe())

MinMaxScaler = MinMaxScaler()
# x = MinMaxScaler.fit_transform(x)
X["pH"] = MinMaxScaler.fit_transform(X["pH"].to_numpy().reshape(-1,1))
# X["Temprature"] = MinMaxScaler.fit_transform(X["Temprature"].to_numpy().reshape(-1,1))
# print(x.loc[:, ["pH","MinMax_pH"]].head())
print(pd.DataFrame(X).describe())
# แบ่งข้อมูลเป็น 80:20 โดยจะใช้ 80% ของข้อมูลไป Train Model และใช้ 20% ของข้อมูลไป Test Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# print("X_train",X_train)
# print("X_test",X_test)0
#
# print("y_train",y_train)
# print("y_test",y_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# # # Decision Tree
dtModel = DecisionTreeClassifier()
startTime = time.time()

dtModel.fit(X_train, np.ravel(y_train))

dtY_pred = dtModel.predict(X_test)
dtScore = round(accuracy_score(y_test, dtY_pred), 2)
dtTime = round(time.time() - startTime, 2)

print("Accuracy using Decision Tree: ",dtScore)
print("Time taken using Decision Tree: ", dtTime)

# from sklearn.svm import SVC
# svmModel = SVC(kernel='linear')
# startTime = time.time()
# # Building the model using the training data set
# svmModel.fit(X_train, np.ravel(y_train))
# # Evaluating the model using testing data set
# svmY_pred = svmModel.predict(X_test)
# svmScore = round(accuracy_score(y_test, svmY_pred), 2)
# svmTime = round(time.time() - startTime, 2)
# # Printing the accuracy and the time taken by the classifier
# print('Accuracy using Support Vector Machine:', svmScore)
# print('Time taken using Support Vector Machine:', svmTime)


# Neural Networks
# from sklearn.neural_network import MLPClassifier
# nnModel = MLPClassifier()
# startTime = time.time()
# # Building the model using the training data set
# nnModel.fit(X_train, np.ravel(y_train))
# # Evaluating the model using testing data set
# nnY_pred = nnModel.predict(X_test)
# nnScore = round(accuracy_score(y_test, nnY_pred), 2)
# nnTime = round(time.time() - startTime, 2)
# # Printing the accuracy and the time taken by the classifier
# print('Accuracy using Neural Networks:', nnScore)
# print('Time taken using Neural Networks:', nnTime)



# Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# nbModel = GaussianNB()
# startTime = time.time()
# # Building the model using the training data set
# nbModel.fit(X_train, np.ravel(y_train))
# # Evaluating the model using testing data set
# nbY_pred = nbModel.predict(X_test)
# nbScore = round(accuracy_score(y_test, nbY_pred), 2)
# nbTime = round(time.time() - startTime, 2)
# # Printing the accuracy and the time taken by the classifier
# print('Accuracy using Naive Bayes:', nbScore)
# print('Time taken using Naive Bayes:', nbTime)

from sklearn import metrics

# Training and Evaluating
def draw_confusion_matrix(y_test,test_predict):
    confusion_matrix = metrics.confusion_matrix(y_test, test_predict)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

    cm_display.plot()
    plt.show()
draw_confusion_matrix(y_test,dtY_pred)


from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(dtModel,
                           out_file=None,
                           feature_names=X.columns,
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree") # Save the tree as PDF
graph.view() # Show the tree in default viewer (usually a PDF viewer)


importances = dtModel.feature_importances_
print(importances)
indices = np.argsort(importances)
features = ["pH", "Temprature", "Taste", "Odor", "Fat ", "Turbidity","Colour"]

j = 5  # top j importance
fig = plt.figure(figsize=(16, 9))
plt.barh(range(j), importances[indices][len(indices) - j:], color='lightblue', align='center')
plt.yticks(range(j), [features[i] for i in indices[len(indices) - j:]])
plt.title('Feature Importances')
plt.xlabel('Relative Importance')
plt.show()