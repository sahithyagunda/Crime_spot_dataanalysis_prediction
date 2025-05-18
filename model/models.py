import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv('crime_data.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])



X = df.drop(["Risk_levels"], axis=1)
categorical_cols = ["City", "Day_of_Week", "Population_Density", "Area_Type", "Nearby_Facility"]
X_encoded = pd.get_dummies(X, columns=categorical_cols)


encoder = LabelEncoder()
y = encoder.fit_transform(df['Risk_levels'])


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42) 

rfclassifier= RandomForestClassifier(n_estimators=20)
rfclassifier.fit(X_train,y_train)
y_pred = rfclassifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,average='macro')


#print(accuracy)
#print(precision)


with open('rfclassifier.pkl','wb') as file:
    pickle.dump(rfclassifier,file)

with open('encoder.pkl','wb') as file:
    pickle.dump(encoder,file)



