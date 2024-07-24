import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# CSV dosyasını okuyun
data = pd.read_csv('train_cleaned.csv')

data_sample = data.sample(frac=0.1, random_state=42)


#Veri setini özellik ve hedef değişkenlere ayırma:
X = data_sample.drop('Credit_Score', axis=1)
y = data_sample['Credit_Score']

#Sayısal olmayan sütunları kontrol etme ve dönüştürme:

# Sayısal olmayan sütunları tespit edin
non_numeric_columns = X.select_dtypes(include=['object']).columns
# Sayısal olmayan sütunları dummies yöntemiyle dönüştürün
X = pd.get_dummies(X, columns=non_numeric_columns, drop_first=True)

#eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#özellik ölçekleme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model oluşturma ve eğitme
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#test
y_pred = model.predict(X_test)

#performans değerlendirme
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Accuracy Score:')
print(accuracy_score(y_test, y_pred))
