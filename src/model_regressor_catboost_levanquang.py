import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'catboost_regressor.pkl')

def train_catboost_regressor(data_path=DATA_PATH, model_path=MODEL_PATH):
    print("Đang tải dữ liệu cho CatBoost Regressor...")
    df = pd.read_csv(data_path)
    
    # Đặc trưng và Mục tiêu
    # Mục tiêu là delivery_days_capped
    drop_cols = ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                 'order_approved_at', 'order_delivered_timestamp', 'order_estimated_delivery_date', 
                 'delivery_days', 'delivery_days_capped', 'is_extreme', 'delay_vs_estimated', 'delivery_class']
    
    # Kiểm tra xem các cột có tồn tại không
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df['delivery_days_capped'] # Sử dụng mục tiêu đã giới hạn
    
    # Xử lý các biến phân loại
    # Đã thêm seller_id và payment_type
    cat_features = ['product_category_name', 'customer_city', 'customer_state', 'seller_id', 'payment_type']
    
    valid_cat_features = []
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
            valid_cat_features.append(col)
            
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    
    
    # Khởi tạo CatBoost Regressor với các siêu tham số đã cải thiện
    # Mục tiêu: Tăng R2 từ 0.39 lên 0.50+, giảm MAE từ 4.02 xuống 3.50
    reg = CatBoostRegressor(
        iterations=2500,           # Tăng từ 1500 để học tốt hơn
        learning_rate=0.03,        # Giảm từ 0.05 để hội tụ ổn định hơn
        depth=10,                  # Tăng từ 8 để bắt các mẫu phức tạp hơn
        l2_leaf_reg=3,            # Thêm regularization chống overfitting
        subsample=0.8,            # Lấy mẫu ngẫu nhiên 80% dữ liệu mỗi lần lặp
        border_count=254,          # Tăng độ chính xác cho các đặc trưng phân loại
        loss_function='MAE',       # Tối ưu hóa MAE
        cat_features=valid_cat_features,
        verbose=100,
        random_seed=42,
        early_stopping_rounds=150, # Tăng từ 100 để kiên nhẫn hơn
        task_type="CPU"
    )
    
    print("Đang huấn luyện CatBoost Regressor với các siêu tham số đã cải thiện...")
    reg.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=150)
    
    # Đánh giá
    print("\n--- Đánh giá CatBoost Regressor ---")
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
    train_catboost_regressor()
