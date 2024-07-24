import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

# CSV dosyasını okuyun
veri = pd.read_csv('train_cleaned.csv')

# Kategorik ve sayısal sütunları belirleyin
categorical_columns = ['Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
numerical_columns = [col for col in veri.columns if col not in categorical_columns + ['Credit_Score', 'Credit_History_Age']]

# Credit_History_Age sütununu işleyin
def process_credit_history_age(age_str):
    try:
        years, months = age_str.split(' Years and ')
        months = months.replace(' Months', '')
        return int(years) * 12 + int(months)
    except:
        return np.nan
    
veri['Credit_History_Age'] = veri['Credit_History_Age'].apply(process_credit_history_age)

# Hedef ve özellikleri ayırın
X = veri.drop(['Credit_Score'], axis=1)
y = veri['Credit_Score']

# Kategorik sütunları encode edin
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)],
    remainder='drop')

# Kategorik verileri encode edin
X_encoded = preprocessor.fit_transform(X)

# Veri setini eğitim ve test veri seti olarak bölün
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=42)

# SMOTE kullanarak veri setini dengeli hale getirin
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Random Forest modelini oluşturun
rf = RandomForestClassifier(random_state=42)

# Modeli eğitin
rf.fit(X_train_res, y_train_res)

# Modeli test veri seti ile tahmin yapın
y_pred = rf.predict(X_test)

# Modelin performansını değerlendirin
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Accuracy Score:')
print(accuracy_score(y_test, y_pred))
