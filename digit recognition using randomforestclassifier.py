# Step 1: Data Collection
import pandas as pd
sa=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\digit recognition using randomforestclassifier 6th project\train.csv")
print(sa)

# Step 1: Data understanding


print(sa.tail(3))
print(sa.head(5))
print(sa.describe)
print(sa.info)
print(sa.shape)

print(sa.isnull().sum())
#graph 


"""import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
sns.pairplot(sa.reset_index(),palette="husl",hue='y',height=3)
sns.set_style("drakgrid")
plt.title("")
plt.show()"""
#find x and y
x=sa.iloc[:,1:].values
print(x)
y=sa.iloc[:,0].values
print(y)



#train_test_split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#modeling

from sklearn .ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)

#predict of model

pred=model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("accuracy_score of 1st model:-{0}%".format(accuracy_score(y_test,pred)*100))