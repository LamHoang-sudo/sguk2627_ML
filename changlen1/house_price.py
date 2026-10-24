import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

# Nạp dữ liệu
df = pd.read_csv(
    "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
)

print("Nạp dữ liệu thành công!")
print("Số dòng:", df.shape[0])
print("Số cột:", df.shape[1])

display(df.head())
# Xem thông tin các cột
df.info()

# Thống kê các thuộc tính số
display(df.describe())

# Thống kê các thuộc tính phân loại
display(df.describe(include="object"))

# Kiểm tra dòng trùng lặp
print("Số dòng trùng lặp:", df.duplicated().sum())

# Kiểm tra số lượng giá trị thiếu
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("Các cột bị thiếu dữ liệu:")
display(missing.to_frame(name="Số lượng thiếu"))
plt.figure(figsize=(12, 6))

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

sns.barplot(x=missing.index, y=missing.values)

plt.title("Số lượng giá trị thiếu theo từng thuộc tính")
plt.xlabel("Thuộc tính")
plt.ylabel("Số lượng giá trị thiếu")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
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
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Mối quan hệ giữa diện tích và giá nhà")
plt.xlabel("Diện tích sinh hoạt (GrLivArea)")
plt.ylabel("Giá bán (SalePrice)")
plt.show()
# Tạo bản sao để không thay đổi dữ liệu gốc
df_clean = df.copy()

# Đếm số quan sát có diện tích lớn hơn 4000
outlier_count = (df_clean["GrLivArea"] > 4000).sum()

print("Số quan sát GrLivArea > 4000:", outlier_count)

# Loại các quan sát này nếu biểu đồ và phân tích của bạn xác nhận
# chúng là những trường hợp bất thường cần loại bỏ
df_clean = df_clean[df_clean["GrLivArea"] <= 4000].copy()

print("Kích thước sau xử lý outlier:", df_clean.shape)
# Những thuộc tính phân loại mà NaN thường có nghĩa là
# đặc điểm đó không tồn tại trong căn nhà
categorical_none_cols = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC",
    "Fence", "MiscFeature"
]

# Những thuộc tính số mà NaN thường có nghĩa là
# đặc điểm tương ứng không tồn tại
numeric_zero_cols = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",
    "BsmtHalfBath", "GarageCars", "GarageArea"
]

print("Số cột cần xử lý:", df_clean.isnull().any().sum())
print("Số ô dữ liệu thiếu:", df_clean.isnull().sum().sum())

# Ghi nhận các cột thiếu để kiểm tra
display(
    df_clean.isnull().sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
    .to_frame("Số lượng thiếu")
)
# Tách biến đầu vào và biến mục tiêu
X = df_clean.drop(columns=["SalePrice", "Id"])
y = df_clean["SalePrice"]

# Chia 70% train và 30% phần còn lại
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# Chia phần còn lại thành 15% validation và 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("Kích thước tập Train:", X_train.shape, y_train.shape)
print("Kích thước tập Validation:", X_val.shape, y_val.shape)
print("Kích thước tập Test:", X_test.shape, y_test.shape)

print("\nTỷ lệ phân chia:")
print(f"Train:      {len(X_train) / len(X):.1%}")
print(f"Validation: {len(X_val) / len(X):.1%}")
print(f"Test:       {len(X_test) / len(X):.1%}")
from sklearn.impute import SimpleImputer

# Xác định các cột số và phân loại
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# Chỉ điền 0 hoặc None cho các cột có ý nghĩa vắng mặt
for col in numeric_zero_cols:
    if col in X_train.columns:
        X_train.loc[:, col] = X_train[col].fillna(0)
        X_val.loc[:, col] = X_val[col].fillna(0)
        X_test.loc[:, col] = X_test[col].fillna(0)

for col in categorical_none_cols:
    if col in X_train.columns:
        X_train.loc[:, col] = X_train[col].fillna("None")
        X_val.loc[:, col] = X_val[col].fillna("None")
        X_test.loc[:, col] = X_test[col].fillna("None")

# Cập nhật danh sách cột sau khi xử lý các giá trị vắng mặt
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# Học cách điền từ tập train
numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")

X_train.loc[:, numeric_cols] = numeric_imputer.fit_transform(
    X_train[numeric_cols]
)
X_val.loc[:, numeric_cols] = numeric_imputer.transform(
    X_val[numeric_cols]
)
X_test.loc[:, numeric_cols] = numeric_imputer.transform(
    X_test[numeric_cols]
)

X_train.loc[:, categorical_cols] = categorical_imputer.fit_transform(
    X_train[categorical_cols]
)
X_val.loc[:, categorical_cols] = categorical_imputer.transform(
    X_val[categorical_cols]
)
X_test.loc[:, categorical_cols] = categorical_imputer.transform(
    X_test[categorical_cols]
)

# Kiểm tra kết quả
print("Số giá trị thiếu trong X_train:", X_train.isnull().sum().sum())
print("Số giá trị thiếu trong X_val:", X_val.isnull().sum().sum())
print("Số giá trị thiếu trong X_test:", X_test.isnull().sum().sum())