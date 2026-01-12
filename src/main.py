import os
import sys

# Xác định thư mục gốc tương đối với file này
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

# Import các module
# Giả sử chúng nằm trong cùng thư mục
import preprocessing
import eda
import model_classifier_catboost_levanquang as model_classifier_catboost
import model_regressor_catboost_levanquang as model_regressor_catboost
import model_regressor_xgboost_tranmanhhung as model_regressor_xgboost
import evaluation

def main():
    print("==================================================")
    print("       QUY TRÌNH DỰ ĐOÁN GIAO HÀNG LOGISTICS      ")
    print("==================================================")
    
    # Bước 1: Tiền xử lý
    print("\n[Bước 1] Đang tiền xử lý dữ liệu...")
    if not os.path.exists(DATA_PATH):
        orders, items, products, customers, payments = preprocessing.load_data()
        preprocessing.preprocess_data(orders, items, products, customers, payments).to_csv(DATA_PATH, index=False)
        print("Dữ liệu đã được tiền xử lý và lưu.")
    else:
        print("Đã tìm thấy dữ liệu đã xử lý. Bỏ qua xử lý dữ liệu thô.")
        
    # Bước 2: EDA
    print("\n[Bước 2] Phân tích dữ liệu khám phá (EDA)...")
    eda.run_eda(file_path=DATA_PATH)
    
    # Bước 3: Huấn luyện bộ phân loại Giai đoạn 1
    print("\n[Bước 3] Đang huấn luyện bộ phân loại Giai đoạn 1 (CatBoost)...")
    model_classifier_catboost.train_classifier(data_path=DATA_PATH)
    
    # Bước 4: Huấn luyện bộ hồi quy Giai đoạn 2
    print("\n[Bước 4] Đang huấn luyện các bộ hồi quy Giai đoạn 2...")
    print("  -> Đang huấn luyện CatBoost Regressor...")
    model_regressor_catboost.train_catboost_regressor(data_path=DATA_PATH)
    
    print("  -> Đang huấn luyện XGBoost Regressor...")
    model_regressor_xgboost.train_xgboost_regressor(data_path=DATA_PATH)
    
    # Bước 5: Đánh giá & So sánh
    print("\n[Bước 5] Đang đánh giá & so sánh cuối cùng...")
    evaluation.evaluate_system()
    
    print("\n==================================================")
    print("               QUY TRÌNH HOÀN TẤT                 ")
    print("==================================================")

if __name__ == "__main__":
    main()
