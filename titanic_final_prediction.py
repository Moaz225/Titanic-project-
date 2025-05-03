# 📦 Импорт необходимых библиотек
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📥 Загрузка данных
df = pd.read_csv("E:/data since/titanic.csv")

# 🧠 Feature Engineering
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

# 🧼 Подготовка данных
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
df_model = df.drop(columns=drop_cols)

categorical_cols = df_model.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

imputer = SimpleImputer(strategy="mean")
df_model = pd.DataFrame(imputer.fit_transform(df_model), columns=df_model.columns)

# 🔄 Разделение на обучающую и тестовую выборки
target = "Survived"
X = df_model.drop(columns=[target])
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Обучение финальной модели
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# 🔮 Предсказание
y_pred = model.predict(X_test)

# 📊 Оценка
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("📌 Точность модели:", round(accuracy * 100, 2), "%\n")
print("🧾 Classification Report:\n", report)
print("🔢 Confusion Matrix:\n", conf_matrix)
