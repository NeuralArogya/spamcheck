from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from data_loader import load_data

df = load_data()
encoder = LabelEncoder()
df['label_num'] = encoder.fit_transform(df['label'])

cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, 'spam_classifier.pkl')
joblib.dump(cv, 'vectorizer.pkl')
