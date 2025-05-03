# 📦 Импорт необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 📥 Загрузка данных
df = pd.read_csv("E:/data since/titanic.csv")

# 🧠 Feature Engineering (должен быть выполнен заранее или повторён здесь)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

def age_category(age):
    if pd.isnull(age):
        return "Unknown"
    elif age < 13:
        return "Child"
    elif age < 60:
        return "Adult"
    else:
        return "Senior"

df["AgeGroup"] = df["Age"].apply(age_category)

# 🎯 Целевой признак
target = "Survived"

# 🧼 Удаление ненужных столбцов
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
df_model = df.drop(columns=drop_cols)

# 🔄 Кодирование категориальных переменных
categorical_cols = df_model.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# 🧩 Заполнение пропущенных значений
imputer = SimpleImputer(strategy="mean")
df_model = pd.DataFrame(imputer.fit_transform(df_model), columns=df_model.columns)

# 📦 Разделение на обучающую и тестовую выборки
X = df_model.drop(columns=[target])
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Проверка форм
print("Размер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)
