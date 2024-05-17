from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Загрузка набора данных Breast Cancer
data = load_breast_cancer()

# Разделение датасета на обучающую и тестовую выборки с random_state=42
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Разделение датасета на обучающую и тестовую выборки с random_state=1
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(data.data, data.target, test_size=0.2, random_state=1)

# Проверка размеров обучающих и тестовых выборок
print("Размер обучающей выборки с random_state=50:", X_train_50.shape)
print("Размер тестовой выборки с random_state=50:", X_test_50.shape)
print("Размер обучающей выборки с random_state=1:", X_train_1.shape)
print("Размер тестовой выборки с random_state=1:", X_test_1.shape)