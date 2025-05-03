# 📦 Импорт необходимых библиотек
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 📥 Загрузка и подготовка данных
df = pd.read_csv("E:/data since/titanic.csv")
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

drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
df_model = df.drop(columns=drop_cols)
categorical_cols = df_model.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

imputer = SimpleImputer(strategy="mean")
df_model = pd.DataFrame(imputer.fit_transform(df_model), columns=df_model.columns)

# 🎯 Разделение данных
target = "Survived"
X = df_model.drop(columns=[target])
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧪 Модели для тестирования
models = {
    "Логистическая регрессия": LogisticRegression(max_iter=1000),
    "Дерево решений": DecisionTreeClassifier(),
    "Градиентный бустинг": GradientBoostingClassifier(),
    "Нейронная сеть": MLPClassifier(max_iter=1000)
}

# 📈 Оценка моделей с кросс-валидацией
print("Результаты кросс-валидации (5 фолдов):")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: средняя точность = {scores.mean():.4f}")
