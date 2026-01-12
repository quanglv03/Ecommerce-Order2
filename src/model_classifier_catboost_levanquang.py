import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'catboost_classifier.pkl')

def train_classifier(data_path=DATA_PATH, model_path=MODEL_PATH):
    print("Đang tải dữ liệu cho bộ phân loại...")
    df = pd.read_csv(data_path)
    
    # Đặc trưng và Mục tiêu
    # Loại bỏ các cột liên quan đến mục tiêu và rò rỉ dữ liệu
    drop_cols = ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
                 'order_approved_at', 'order_delivered_timestamp', 'order_estimated_delivery_date', 
                 'delivery_days', 'delivery_days_capped', 'is_extreme', 'delay_vs_estimated', 'delivery_class']
    
    # Kiểm tra xem các cột có tồn tại trước khi loại bỏ không
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df['delivery_class']
    
    # Xử lý các biến phân loại
    # CatBoost tự động xử lý chúng nếu được chỉ định
    cat_features = ['product_category_name', 'customer_city', 'customer_state', 'seller_id', 'payment_type']
    
    # Đảm bảo chúng là chuỗi và tồn tại trong X
    valid_cat_features = []
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str)
            valid_cat_features.append(col)
            
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Phân phối lớp trong tập huấn luyện:\n{y_train.value_counts(normalize=True)}")
    print(f"Số lượng lớp: {y_train.value_counts().to_dict()}")
    
    # ===== TÍNH TOÁN TRỌNG SỐ LỚP TÙY CHỈNH =====
    # Phạt nặng việc phân loại sai các lớp thiểu số
    class_counts = y_train.value_counts().sort_index()
    total_samples = len(y_train)
    n_classes = len(class_counts)
    
    # Tính toán trọng số cân bằng với tỷ lệ bổ sung cho lớp cực trị
    class_weights = {
        0: 1.0,                                                    # Bình thường (lớp đa số)
        1: (total_samples / (n_classes * class_counts[1])) * 1.5, # Trễ: tăng 1.5 lần
        2: (total_samples / (n_classes * class_counts[2])) * 2.0  # Cực kỳ trễ: tăng 2.0 lần
    }
    
    print(f"\nSử dụng trọng số lớp tùy chỉnh:")
    for cls, weight in class_weights.items():
        print(f"  Lớp {cls}: {weight:.2f}")
    
    # Khởi tạo CatBoost với các siêu tham số đã cải thiện
    clf = CatBoostClassifier(
        iterations=2000,           # Tăng từ 1000
        learning_rate=0.03,        # Giảm từ 0.05
        depth=10,                  # Tăng từ 8
        l2_leaf_reg=5,            # Thêm regularization
        border_count=254,          # Tăng độ chính xác
        loss_function='MultiClass',
        eval_metric='TotalF1',
        cat_features=valid_cat_features,
        class_weights=list(class_weights.values()),  # Sử dụng trọng số tùy chỉnh
        verbose=100,
        random_seed=42,
        early_stopping_rounds=150,  # Tăng từ 100
        task_type="CPU"
    )
    
    print("\nĐang huấn luyện bộ phân loại CatBoost với các siêu tham số đã cải thiện...")
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=150)
    
    # Đánh giá
    print("\n--- Đánh giá bộ phân loại ---")
    y_pred = clf.predict(X_test)
    
    print("Ma trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred))
    
    recall = recall_score(y_test, y_pred, average='macro')
    print(f"Recall trung bình (Macro): {recall:.4f}")
    
    # Lưu mô hình
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Mô hình đã được lưu tại {model_path}")
    
    return clf

if __name__ == "__main__":
    train_classifier()
