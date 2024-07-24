import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

# CSV dosyasını okuyun
veri = pd.read_csv('train_cleaned.csv')

# Kategorik ve sayısal sütunları belirleyin
categorical_columns = ['Month', 'Name', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
numerical_columns = [col for col in veri.columns if col not in categorical_columns + ['Credit_Score', 'ID', 'SSN', 'Credit_History_Age']]

# Credit_History_Age sütununu işleyin
def process_credit_history_age(age_str):
    try:
        years, months = age_str.split(' Years and ')
        months = months.replace(' Months', '')
        return int(years) * 12 + int(months)
    except:
        return np.nan

veri['Credit_History_Age'] = veri['Credit_History_Age'].apply(process_credit_history_age)

# Hedef ve özellikleri ayırın, ID ve SSN sütunlarını çıkarın
X = veri.drop(['Credit_Score', 'ID', 'SSN'], axis=1)
y = veri['Credit_Score']

# Verinin %10'unu eğitim, diğer %10'unu test verisi olarak seçin
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.0001, random_state=42)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=0.0001, random_state=42)

# Kategorik sütunları encode edin
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],
    remainder='drop')

# Kategorik verileri encode edin
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# SMOTE kullanarak veri setini dengeli hale getirin
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_encoded, y_train)

# Gradient Boosting model için parametre aralığı belirleyin
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Grid Search kullanarak en iyi parametreleri bulun
grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

# En iyi modeli ve parametreleri alın
best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# Modeli test veri seti ile tahmin yapın
y_pred = best_model.predict(X_test_encoded)

# Modelin performansını değerlendirin
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Accuracy Score:')
print(accuracy_score(y_test, y_pred))
