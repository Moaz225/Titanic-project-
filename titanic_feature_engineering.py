# 📦 Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Загрузка данных
df = pd.read_csv("E:/data since/titanic.csv")

# 🧠 Feature Engineering

# 🎩 Извлечение титула из имени
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

# 👨‍👩‍👧‍👦 Создание признака FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 👶 Категоризация возраста
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

# 📊 Визуализация: выживаемость по Title
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Title", hue="Survived", order=df["Title"].value_counts().index)
plt.title("Выживаемость по титулу")
plt.xlabel("Титул")
plt.ylabel("Количество пассажиров")
plt.legend(title="Выжил", labels=["Нет", "Да"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📊 Визуализация: выживаемость по AgeGroup
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="AgeGroup", hue="Survived", order=["Child", "Adult", "Senior", "Unknown"])
plt.title("Выживаемость по возрастной группе")
plt.xlabel("Возрастная группа")
plt.ylabel("Количество пассажиров")
plt.legend(title="Выжил", labels=["Нет", "Да"])
plt.tight_layout()
plt.show()

# 📊 Визуализация: выживаемость по FamilySize
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="FamilySize", hue="Survived")
plt.title("Выживаемость по размеру семьи")
plt.xlabel("Размер семьи")
plt.ylabel("Количество пассажиров")
plt.legend(title="Выжил", labels=["Нет", "Да"])
plt.tight_layout()
plt.show()
