import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('drug200 (1).csv')

print(data.head())
data.dropna(inplace=True)
X = data.drop('Drug', axis=1)
y = data['Drug']

le_sex = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'])
le_bp = LabelEncoder()
X['BP'] = le_bp.fit_transform(X['BP'])
le_cholesterol = LabelEncoder()
X['Cholesterol'] = le_cholesterol.fit_transform(X['Cholesterol'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

classifier = DecisionTreeClassifier(random_state=25)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
