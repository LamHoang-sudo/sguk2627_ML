import numpy as np
import pandas as pd

# ============================================================
# 2. LOAD DATA
# ============================================================
train = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/test.csv')

print("Train Shape gốc:", train.shape)
print("Test Shape gốc:", test.shape)

# Lưu ID để tạo file submission
test_id = test["Id"].copy()

# Lưu SalePrice và chuyển sang log
y_train = np.log1p(train["SalePrice"])

# Không sử dụng Id và SalePrice làm feature
train_features = train.drop(columns=["Id", "SalePrice"])
test_features = test.drop(columns=["Id"])

# ============================================================
# 3. COMBINE TRAIN AND TEST FOR CONSISTENT ENCODING
# ============================================================
# Lưu lại số dòng của train để lát sau cắt ra cho chuẩn
n_train = len(train_features)

# Gộp chung train và test lại thành 1 tập all_data
all_data = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# ============================================================
# 4. FEATURE ENGINEERING & XỬ LÝ DỮ LIỆU
# ============================================================
# Điền khuyết LotFrontage bằng median của khu vực (Neighborhood)
if 'LotFrontage' in all_data.columns:
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Xử lý nhanh Missing Values ban đầu
num_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[num_cols] = all_data[num_cols].fillna(0)

cat_cols = all_data.select_dtypes(include=['object']).columns
all_data[cat_cols] = all_data[cat_cols].fillna('None')

# Tạo các đặc trưng mới (Feature Engineering)
bsmt_sf = all_data['TotalBsmtSF'] if 'TotalBsmtSF' in all_data.columns else 0
flr1_sf = all_data['1stFlrSF'] if '1stFlrSF' in all_data.columns else 0
flr2_sf = all_data['2ndFlrSF'] if '2ndFlrSF' in all_data.columns else 0
all_data['TotSF'] = bsmt_sf + flr1_sf + flr2_sf

full_bath = all_data['FullBath'] if 'FullBath' in all_data.columns else 0
half_bath = all_data['HalfBath'] if 'HalfBath' in all_data.columns else 0
bsmt_full = all_data['BsmtFullBath'] if 'BsmtFullBath' in all_data.columns else 0
bsmt_half = all_data['BsmtHalfBath'] if 'BsmtHalfBath' in all_data.columns else 0
all_data['TotalBath'] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

if 'YrSold' in all_data.columns and 'YearBuilt' in all_data.columns:
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['HouseAge'] = all_data['HouseAge'].apply(lambda x: max(0, x))

if 'YrSold' in all_data.columns and 'YearRemodAdd' in all_data.columns:
    all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
    all_data['RemodAge'] = all_data['RemodAge'].apply(lambda x: max(0, x))

print('Tạo biến mới thành công!')

# ============================================================
# 5. CHUẨN HOÁ MISSING VALUES LẦN CUỐI & ONE-HOT ENCODING
# ============================================================
# Phân loại lại cột tự động sau khi thêm biến mới
cat_cols_all = all_data.select_dtypes(include=["object", "category"]).columns
num_cols_all = all_data.select_dtypes(include=["number", "bool"]).columns

# Đảm bảo không còn NaN ở cột categorical
for col in cat_cols_all:
    all_data[col] = all_data[col].fillna("Missing").astype(str)

# Đảm bảo không còn NaN ở cột numerical
for col in num_cols_all:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# One-Hot Encoding cho toàn bộ tập dữ liệu gộp
all_data = pd.get_dummies(all_data, columns=list(cat_cols_all), drop_first=False, dtype=int)

# ============================================================
# 6. TÁCH LẠI THÀNH TRAIN VÀ TEST ĐỂ CHUẨN BỊ MODELING
# ============================================================
X_train = all_data.iloc[:n_train].copy()
X_test = all_data.iloc[n_train:].copy()

print("X_train Shape (sau xử lý):", X_train.shape)
print("X_test Shape (sau xử lý):", X_test.shape)

# ============================================================
# 5. FEATURE SELECTION VỚI XGBOOST
# ============================================================
from sklearn.feature_selection import SelectFromModel

print("--- BẮT ĐẦU FEATURE SELECTION CHO XGBOOST ---")
# Huấn luyện mô hình cơ sở để trích xuất độ quan trọng của các đặc trưng
base_xgb = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist")
base_xgb.fit(X_train, y_train)

# Sử dụng SelectFromModel để giữ lại các đặc trưng có độ quan trọng >= mức trung vị (median)
selector = SelectFromModel(base_xgb, threshold='median', prefit=True)
X_train_xgb_selected = selector.transform(X_train)
X_test_xgb_selected = selector.transform(X_test)

print(f"Số lượng đặc trưng ban đầu: {X_train.shape[1]}")
print(f"Số lượng đặc trưng sau khi chọn lọc: {X_train_xgb_selected.shape[1]}")

# ============================================================
# 6. HUẤN LUYỆN VÀ TỐI ƯU SIÊU THAM SỐ TRÊN TẬP ĐẶC TRƯNG MỚI
# ============================================================
print("\n--- BẮT ĐẦU TỐI ƯU SIÊU THAM SỐ ---")
xgb_model_selected = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    tree_method="hist",
    n_jobs=-1
)

param_dist = {
    "n_estimators": [500, 800, 1000, 1500, 2000],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
    "min_child_weight": [1, 2, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.01, 0.05, 0.1, 0.2],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
    "reg_lambda": [1, 2, 5, 10]
}

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
random_cv_xgb = RandomizedSearchCV(
    estimator=xgb_model_selected,
    param_distributions=param_dist,
    n_iter=40,
    scoring="neg_root_mean_squared_error",
    cv=kf,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    return_train_score=True,
    refit=True
)

random_cv_xgb.fit(X_train_xgb_selected, y_train)

print("\nBest Parameters:")
print(random_cv_xgb.best_params_)
print(f"Best CV RMSE (log scale): {-random_cv_xgb.best_score_:.5f}")

# ============================================================
# 7. DỰ ĐOÁN, LƯU MÔ HÌNH VÀ TẠO FILE SUBMISSION
# ============================================================
best_xgb_model = random_cv_xgb.best_estimator_

# Dự đoán trên tập test (đã qua feature selection)
pred_log_xgb = best_xgb_model.predict(X_test_xgb_selected)

# Đưa từ log về giá thực và chặn giá trị âm
predictions_xgb = np.expm1(pred_log_xgb)
predictions_xgb = np.maximum(predictions_xgb, 0)

# Lưu mô hình
with open("xgb_model_selected.pkl", "wb") as f:
    pickle.dump(best_xgb_model, f)
print("\nĐã lưu mô hình: xgb_model_selected.pkl")

# Tạo file Kaggle submission đúng tên hệ thống yêu cầu
submission_xgb = pd.DataFrame({
    "Id": test_id,
    "SalePrice": predictions_xgb
})

# Đổi tên tệp thành submission.csv
submission_xgb.to_csv("submission.csv", index=False)

print("Đã tạo tệp submission.csv thành công!")
print(submission_xgb.head())