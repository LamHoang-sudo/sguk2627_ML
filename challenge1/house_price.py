

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

RANDOM_STATE = 42


def show(obj):

    try:
        display(obj)  
    except NameError:
        print(obj)


CANDIDATE_PATHS = [
    "train.csv",
    "/content/train (1).csv",
]
data_path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError(
        "Không tìm thấy train.csv. Hãy đặt file cùng thư mục với file .py "
        "hoặc sửa lại CANDIDATE_PATHS."
    )

df = pd.read_csv(data_path)

print("Nạp dữ liệu thành công!")
print("Số dòng:", df.shape[0])
print("Số cột:", df.shape[1])
show(df.head())



df.info()

print("\nThống kê các thuộc tính số:")
show(df.describe())

print("\nThống kê các thuộc tính phân loại:")
show(df.describe(include="object"))


print("\nSố dòng trùng lặp:", df.duplicated().sum())


missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nCác cột bị thiếu dữ liệu:")
show(missing.to_frame(name="Số lượng thiếu"))

plt.figure(figsize=(12, 6))
sns.barplot(x=missing.index, y=missing.values)
plt.title("Số lượng giá trị thiếu theo từng thuộc tính")
plt.xlabel("Thuộc tính")
plt.ylabel("Số lượng giá trị thiếu")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


print("\nSkewness của SalePrice:", round(df["SalePrice"].skew(), 2))
print("Kurtosis của SalePrice:", round(df["SalePrice"].kurt(), 2))

plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="SalePrice", kde=True)
plt.title("Phân phối giá bán nhà")
plt.xlabel("SalePrice")
plt.ylabel("Tần suất")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=df["SalePrice"])
plt.title("Boxplot giá bán nhà")
plt.xlabel("SalePrice")
plt.show()


print(
    "\nTương quan GrLivArea - SalePrice:",
    round(df["GrLivArea"].corr(df["SalePrice"]), 3),
)

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
plt.title("Mối quan hệ giữa diện tích và giá nhà")
plt.xlabel("Diện tích sinh hoạt (GrLivArea)")
plt.ylabel("Giá bán (SalePrice)")
plt.show()



df_clean = df.copy()


outliers = df_clean[df_clean["GrLivArea"] > 4000]
print("\nSố quan sát GrLivArea > 4000:", len(outliers))
show(outliers[["Id", "GrLivArea", "SalePrice", "SaleCondition"]])

df_clean = df_clean[df_clean["GrLivArea"] <= 4000].copy()
print("Kích thước sau khi loại outlier:", df_clean.shape)


categorical_none_cols = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC",
    "Fence", "MiscFeature",
]
numeric_zero_cols = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",
    "BsmtHalfBath", "GarageCars", "GarageArea",
]

print("\nTrước khi xử lý thiếu:")
print("Số cột thiếu:", df_clean.isnull().any().sum())
print("Số ô thiếu:", df_clean.isnull().sum().sum())

df_clean[categorical_none_cols] = df_clean[categorical_none_cols].fillna("None")
df_clean[numeric_zero_cols] = df_clean[numeric_zero_cols].fillna(0)


feature_cols = [c for c in df_clean.columns if c not in ("Id", "SalePrice")]
numeric_cols = (
    df_clean[feature_cols].select_dtypes(include=np.number).columns.tolist()
)
categorical_cols = (
    df_clean[feature_cols].select_dtypes(exclude=np.number).columns.tolist()
)

print("\nCác cột còn thiếu (thiếu thật sự):")
still_missing = df_clean[feature_cols].isnull().sum()
show(still_missing[still_missing > 0].to_frame("Số lượng thiếu"))

numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols])
df_clean[categorical_cols] = categorical_imputer.fit_transform(
    df_clean[categorical_cols]
)

print("\nSau khi xử lý thiếu:")
print("Số ô thiếu trong df_clean:", df_clean.isnull().sum().sum())



X = df_clean.drop(columns=["SalePrice", "Id"])
y = df_clean["SalePrice"]


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
)

print("\nKích thước tập Train:     ", X_train.shape, y_train.shape)
print("Kích thước tập Validation:", X_val.shape, y_val.shape)
print("Kích thước tập Test:      ", X_test.shape, y_test.shape)

print("\nTỷ lệ phân chia:")
print(f"Train:      {len(X_train) / len(X):.1%}")
print(f"Validation: {len(X_val) / len(X):.1%}")
print(f"Test:       {len(X_test) / len(X):.1%}")