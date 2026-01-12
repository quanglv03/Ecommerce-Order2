import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_regressor.pkl')

def train_xgboost_regressor(data_path=DATA_PATH, model_path=MODEL_PATH):
    print("Đang tải dữ liệu cho XGBoost Regressor...")
    df = pd.read_csv(data_path)
    
    # Đặc trưng và Mục tiêu
    drop_cols = ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                 'order_approved_at', 'order_delivered_timestamp', 'order_estimated_delivery_date', 
                 'delivery_days', 'delivery_days_capped', 'is_extreme', 'delay_vs_estimated', 'delivery_class']
    
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df['delivery_days_capped']
    
    # Xử lý các biến phân loại cho XGBoost
    # Chuyển đổi cột object sang kiểu category
    # Đã thêm seller_id và payment_type
    cat_features = ['product_category_name', 'customer_city', 'customer_state', 'seller_id', 'payment_type']
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    
    
    # Khởi tạo XGBoost Regressor với các siêu tham số đã cải thiện
    # Mục tiêu: Cải thiện R2 và giảm MAE để cạnh tranh với CatBoost
    reg = xgb.XGBRegressor(
        n_estimators=2500,         # Tăng từ 1500 để học tốt hơn
        learning_rate=0.03,        # Giảm từ 0.05 để hội tụ ổn định hơn
        max_depth=10,              # Tăng từ 8 để bắt các mẫu phức tạp hơn
        min_child_weight=3,        # Thêm để chống overfitting
        subsample=0.8,             # Lấy mẫu ngẫu nhiên 80% dữ liệu
        colsample_bytree=0.8,      # Lấy mẫu ngẫu nhiên 80% đặc trưng
        gamma=0.1,                 # Giảm thiểu loss tối thiểu để chia tách
        reg_alpha=0.1,             # L1 regularization
        reg_lambda=1.0,            # L2 regularization
        objective='reg:absoluteerror',  # Tối ưu hóa MAE
        tree_method='hist',
        enable_categorical=True,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=150  # Tăng từ 100 để kiên nhẫn hơn
    )
    
    print("Đang huấn luyện XGBoost Regressor với các siêu tham số đã cải thiện...")
    reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # Đánh giá
    print("\n--- Đánh giá XGBoost Regressor ---")
    y_pred = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Lưu mô hình
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(model_path, 'wb') as f:
        pickle.dump(reg, f)
    print(f"Mô hình đã được lưu tại {model_path}")
    
    return reg

if __name__ == "__main__":
    train_xgboost_regressor()
