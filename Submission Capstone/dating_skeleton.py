import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score
import time

sns.set()

#Create your df here:

df = pd.read_csv(r'C:\Users\xavier\MyScripts\profiles.csv')


# Exploratory Data Analysis
#print (df.status.head())
#print (df.body_type.unique())
#print (df.diet.value_counts())

# Create new variables

df.dropna(axis=0,subset=["age","diet","sex","education"],inplace=True)

# # New mapping for bodytype
df["body_type"] = df["body_type"].fillna(value="rather not say")
body_type_mapping = {"used up": 0,"skinny": 1, "thin": 2, "athletic": 3, "fit": 4, "average": 5, "rather not say":6, "jacked": 7, "a little extra": 8, "curvy": 9,"full figured": 10, "overweight": 11,}
df["body_type_code"] = df.body_type.map(body_type_mapping)

# Checking if it worked
#print (df.body_type_code.value_counts())


## New binary variable to describe people with vegan/vegetarian diet
df["veg_code"]=np.where(df["diet"].str.contains("veg"),1,0)
# Checking if variable was created
print (df.veg_code.value_counts())

veg = df[df.veg_code == 1]
non_veg = df[df.veg_code == 0]

# Graph to explore link  between veg diet and bodytpe
#= plt.hist([veg.body_type_code,non_veg.body_type_code],color=["green","blue"],density=True,histtype="bar",alpha=0.4)
# = plt.legend(("vegetarians/vegans","other diet"), loc='upper right')
# = plt.xlabel("Bodytype")
# = plt.ylabel("Frequency")
# = plt.title("Bodytype by diet")
# = plt.show()

#  Graph to explore link  between veg diet and age
#_= sns.boxplot(x='veg_code', y='age', data=df)
#_= plt.xlabel('Is vegetarian/vegan')
#_= plt.ylabel('Age')
#_= plt.title('Distribution of age by diet category')
#= plt.show()

# Graph to explore link between veg diet and gender
_=plt.hist([veg.sex,non_veg.sex],color=["green","gray"],bins=2,histtype="bar", stacked=True, alpha =0.9)
_= plt.xlabel('Gender',fontsize=8)
_= plt.xticks(fontsize=8)
_= plt.ylabel('Number of people', fontsize=8)
_= plt.yticks(fontsize=8)
_ =plt.legend(("vegetarians/vegans","other diet"), loc='upper right',fontsize=8)
_= plt.title('Gender by diet category')
_= plt.show()
# There are less girls than boys in the dating data, but more vegs among girls than boys

df["sex_code"]=np.where(df["sex"].str.contains("m"),1.0,2.0)
# Checking if it worked
#print (df.sex_code.value_counts())

#Education
education_mapping ={"working on college/university":0,
"working on space camp":1,
"graduated from masters program":2,
"graduated from college/university":3,
"working on two-year college":4,
"graduated from high school":5,
"working on masters program":6,
"graduated from space camp":7,
"college/university":8,
"dropped out of space camp":9,
"graduated from ph.d program" :10,
"graduated from law school":11,
"working on ph.d program":12,
"two-year college":13,
"graduated from two-year college" :14,
"working on med school":15,
"dropped out of college/university" :16,
"space camp":17,
"graduated from med school" :18,
"dropped out of high school" :19,
"working on high school" :20,
"masters program" :21,
"dropped out of ph.d program" :22,
"dropped out of two-year college" :23,
"dropped out of med school" :24,
"high school" :25,
"working on law school" :26,
"law school" :27,
"dropped out of masters program" :28,
"ph.d program":29,
"dropped out of law school" :30,
"med school":31}
df["education_code"] = df.education.map(education_mapping)


# Classification approaches
#Can we predict if someone is vegeterian/vegan by his education/gender/age?'

feature_data = df[['education_code', 'age', 'sex_code']]
feature_labels = df["veg_code"]

#Normalisation
x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

X_train, X_test, y_train, y_test  = train_test_split(feature_data, feature_labels, test_size = 0.18, random_state = 20)
# We chose a training set of 72% to get enough "veg" diets in our training sample.
# Only 19% of people have a "veg" diet.


##Naive Bayes
#start = time.time()
NBclassifier = MultinomialNB()
NBclassifier.fit(X_train,y_train)
print (NBclassifier.score(X_test,y_test))
y_predict = NBclassifier.predict(X_test)
print ("Naive Bayes: Accuracy %.2f Recall %.2f Precision %.2f" %(accuracy_score(y_test,y_predict), recall_score(y_test,y_predict),precision_score(y_test,y_predict)))
#end = time.time()
#print(end - start)
# This was discussed with Sen moderator. Naive Bayes Multinomial is not a good model for binary variables.
# Running time : 0.02s
#

##SVM

SVclassifier = SVC(kernel='linear', class_weight={1: 6})
# I use class-weight because it is suppose to yield better results with unblanced sample
SVclassifier.fit(X_train,y_train)
y_predict = SVclassifier.predict(X_test)
print ("SVM: Accuracy %.2f Recall %.2f Precision %.2f" % (accuracy_score(y_test,y_predict), recall_score(y_test,y_predict),precision_score(y_test,y_predict)))
#end = time.time()
#print(end - start)
# Running time: 14s

## Regression approaches
#Can we guess age  observing diet(vegetarian/vegan), gender and bodytype?
feature_data2 = df[['veg_code','sex_code', 'body_type_code']]
y = df['age']

x = feature_data2.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data2 = pd.DataFrame(x_scaled, columns=feature_data2.columns)

X_train2, X_test2, y_train2, y_test2  = train_test_split(feature_data2, y, test_size = 0.2, random_state = 1)

#LinearRegression
lm = LinearRegression()
model = lm.fit(X_train2, y_train2)
print ('Linear Regression: Score %.2f' % (model.score(X_test2,y_test2)))
# These features are very poor to predict vegetarian/vegan diet.
# Running time:0.02 sec

#K-nearest
regressor = KNeighborsRegressor(n_neighbors = 5,weights = "distance")
regressor.fit(X_train2,y_train2)
print ('KNN: Score %.2f' % (regressor.score(X_test2,y_test2)))
# This model has a negative score. It is worse than using an arbitrary method to determine age.
# Running time:0.88s
