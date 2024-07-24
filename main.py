import pandas as pd
import numpy as np

veri = pd.read_csv('train.csv')

# Yaşı düzenle
veri['Age'] = veri['Age'].str.replace('-', '').str.replace('_', '')
veri['Age'] = pd.to_numeric(veri['Age'], errors='coerce')

# Yaşı 115'ten büyük olan kayıtları sil
veri = veri[veri['Age'] <= 115]

# Age sütunundaki aykırı değerleri düzenleme (3 sigma kuralı)
mean_age = veri['Age'].mean()
std_age = veri['Age'].std()
max_age = mean_age + 3 * std_age
veri['Age'] = veri['Age'].apply(lambda x: mean_age if x > max_age else x)

# Occupation düzenle
veri['Occupation'] = veri['Occupation'].replace("_______", pd.NA)
mode_occupation = veri['Occupation'].mode()[0]
veri['Occupation'].fillna(mode_occupation, inplace=True)

# Annual_Income düzenle
veri['Annual_Income'] = veri['Annual_Income'].str.replace('-', '').str.replace('_', '')
veri['Annual_Income'] = pd.to_numeric(veri['Annual_Income'], errors='coerce')

# Monthly_Inhand_Salary düzenle
mean_salary = veri['Monthly_Inhand_Salary'].mean()
veri['Monthly_Inhand_Salary'].fillna(mean_salary, inplace=True)

# Num_Bank_Accounts düzenle
veri['Num_Bank_Accounts'] = veri['Num_Bank_Accounts'].astype(str).str.replace('-', '').str.replace('_', '')
veri['Num_Bank_Accounts'] = pd.to_numeric(veri['Num_Bank_Accounts'], errors='coerce')
nb_mean = veri['Num_Bank_Accounts'].mean()
veri['Num_Bank_Accounts'].fillna(nb_mean, inplace=True)

# Num_Credit_Card düzenle
ncc_mean = veri['Num_Credit_Card'].mean()
ncc_std = veri['Num_Credit_Card'].std()
max_ncc = ncc_mean + 3 * ncc_std
veri['Num_Credit_Card'] = veri['Num_Credit_Card'].apply(lambda x: ncc_mean if x > max_ncc else x)
veri['Num_Credit_Card'] = veri['Num_Credit_Card'].astype(int)

# Interest_Rate düzenle
ir_mean = veri['Interest_Rate'].mean()
ir_std = veri['Interest_Rate'].std()
max_ir = ir_mean + 3 * ir_std
veri['Interest_Rate'] = veri['Interest_Rate'].apply(lambda x: ir_mean if x > max_ir else x)

# Num_of_Loan düzenle
veri['Num_of_Loan'] = veri['Num_of_Loan'].str.replace('-', '').str.replace('_', '')
veri['Num_of_Loan'] = pd.to_numeric(veri['Num_of_Loan'], errors='coerce')
veri['Num_of_Loan'] = veri['Num_of_Loan'].astype(int)

# Type_of_Loan düzenle
mode_Type_of_Loan = veri['Type_of_Loan'].mode()[0]
veri['Type_of_Loan'].fillna(mode_Type_of_Loan, inplace=True)

# Delay_from_due_date düzenle
veri['Delay_from_due_date'] = veri['Delay_from_due_date'].astype(str).str.replace('-', '').str.replace('_', '')
veri['Delay_from_due_date'] = pd.to_numeric(veri['Delay_from_due_date'], errors='coerce')
veri['Delay_from_due_date'] = veri['Delay_from_due_date'].astype(int)

# Num_of_Delayed_Payment düzenle
veri['Num_of_Delayed_Payment'] = veri['Num_of_Delayed_Payment'].str.replace('-', '').str.replace('_', '')
veri['Num_of_Delayed_Payment'] = pd.to_numeric(veri['Num_of_Delayed_Payment'], errors='coerce')
ndp_mean = veri['Num_of_Delayed_Payment'].mean()
veri['Num_of_Delayed_Payment'].fillna(ndp_mean, inplace=True)
veri['Num_of_Delayed_Payment'] = veri['Num_of_Delayed_Payment'].astype(int)

# Changed_Credit_Limit düzenle
veri['Changed_Credit_Limit'] = veri['Changed_Credit_Limit'].str.replace('-', '').str.replace('_', '')
veri['Changed_Credit_Limit'] = pd.to_numeric(veri['Changed_Credit_Limit'], errors='coerce')
ccl_mean = veri['Changed_Credit_Limit'].mean()
veri['Changed_Credit_Limit'].fillna(ccl_mean, inplace=True)

# Num_Credit_Inquiries düzenle
veri['Num_Credit_Inquiries'] = pd.to_numeric(veri['Num_Credit_Inquiries'], errors='coerce')
nci_mean = veri['Num_Credit_Inquiries'].mean()
veri['Num_Credit_Inquiries'].fillna(nci_mean, inplace=True)
nci_std = veri['Num_Credit_Inquiries'].std()
max_nci = nci_mean + 3 * nci_std
veri['Num_Credit_Inquiries'] = veri['Num_Credit_Inquiries'].apply(lambda x: nci_mean if x > max_nci else x)
veri['Num_Credit_Inquiries'] = veri['Num_Credit_Inquiries'].astype(int)

# Credit_Mix düzenle
veri['Credit_Mix'] = veri['Credit_Mix'].astype(str).str.replace('-', ' ').str.replace('_', ' ')
veri['Credit_Mix'].fillna('Standard', inplace=True)
veri['Credit_Mix'] = veri['Credit_Mix'].astype('category')

# Outstanding_Debt düzenle
veri['Outstanding_Debt'] = veri['Outstanding_Debt'].astype(str).str.replace('-', '').str.replace('_', '')
veri['Outstanding_Debt'] = pd.to_numeric(veri['Outstanding_Debt'], errors='coerce')
od_mean = veri['Outstanding_Debt'].mean()
veri['Outstanding_Debt'].fillna(od_mean, inplace=True)

# Credit_History_Age düzenle
veri['Credit_History_Age'].fillna('15 Years and 11 Months', inplace=True)

# Amount_invested_monthly düzenle
veri['Amount_invested_monthly'] = pd.to_numeric(veri['Amount_invested_monthly'], errors='coerce')
veri['Amount_invested_monthly'].fillna(veri['Amount_invested_monthly'].mean(), inplace=True)

# Payment_Behaviour düzenle
veri['Payment_Behaviour'] = veri['Payment_Behaviour'].replace("!@9#%8", pd.NA)
mode_Payment_Behaviour = veri['Payment_Behaviour'].mode()[0]
veri['Payment_Behaviour'].fillna(mode_Payment_Behaviour, inplace=True)

# Monthly_Balance düzenle
veri['Monthly_Balance'] = veri['Monthly_Balance'].astype(str).str.replace('-', '').str.replace('_', '')
veri['Monthly_Balance'] = veri['Monthly_Balance'].astype(float)
mb_mean = veri['Monthly_Balance'].mean()
mb_std = veri['Monthly_Balance'].std()
max_mb = mb_mean + 3 * mb_std
veri['Monthly_Balance'] = veri['Monthly_Balance'].apply(lambda x: mb_mean if pd.isna(x) or x > max_mb else x)

# Gereksiz sütunları çıkar
veri = veri.drop(['ID', 'Month', 'Name', 'SSN','Customer_ID'], axis=1)

# Temizlenmiş veriyi kaydet
veri.to_csv('train_cleaned.csv', index=False)

# Null sayılarını kontrol et
print(veri['Credit_Mix'].isnull().sum())
