# 📊 Статистическое описание данных + количество пропущенных значений
summary = df.describe(include='all').T
missing = df.isnull().sum()
summary["Missing"] = missing

import ace_tools as tools; tools.display_dataframe_to_user(name="📋 Статистика и пропущенные значения", dataframe=summary)
