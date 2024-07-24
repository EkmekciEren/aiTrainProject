import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# CSV dosyasını okuyun
veri = pd.read_csv('train_cleaned.csv')

# Eksik değerleri doldurun veya çıkarın
veri = veri.dropna()  # Eksik verileri tamamen çıkarmak için

# Sayısal olmayan sütunları tespit edin ve dummies yöntemiyle dönüştürün
non_numeric_columns = veri.select_dtypes(include=['object']).columns
veri = pd.get_dummies(veri, columns=non_numeric_columns, drop_first=True)

# Hedef ve özellikleri ayırın
X = veri.drop('Credit_Score', axis=1)
y = veri['Credit_Score']

# Veri setini eğitim ve test veri seti olarak bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE kullanarak veri setini dengeli hale getirin
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Özellikleri ölçeklendirin
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Gradient Boosting model için parametre aralığı belirleyin
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Grid Search kullanarak en iyi parametreleri bulun
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

# En iyi modeli ve parametreleri alın
best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# Modeli test veri seti ile tahmin yapın
y_pred = best_model.predict(X_test)

# Modelin performansını değerlendirin
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Accuracy Score:')
print(accuracy_score(y_test, y_pred))
