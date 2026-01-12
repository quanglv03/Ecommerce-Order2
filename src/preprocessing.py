import pandas as pd
import numpy as np
import os
import features

# Xác định thư mục gốc tương đối với file này
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_data(data_dir=DATA_DIR):
    """Tải dữ liệu từ các file CSV."""
    orders = pd.read_csv(os.path.join(data_dir, 'df_Orders.csv'))
    items = pd.read_csv(os.path.join(data_dir, 'df_OrderItems.csv'))
    products = pd.read_csv(os.path.join(data_dir, 'df_Products.csv'))
    customers = pd.read_csv(os.path.join(data_dir, 'df_Customers.csv'))
    payments = pd.read_csv(os.path.join(data_dir, 'df_Payments.csv'))
    return orders, items, products, customers, payments

def preprocess_data(orders, items, products, customers, payments):
    """
    Tiền xử lý dữ liệu: gộp, làm sạch, tạo đặc trưng.
    """
    print("Bắt đầu tiền xử lý...")
    
    # 1. Gộp dữ liệu
    # Gộp Items và Products
    items_products = items.merge(products, on='product_id', how='left')
    
    # Tổng hợp items theo từng đơn hàng (tổng trọng lượng, giá, phí vận chuyển; kích thước lớn nhất)
    # Quan trọng: Giữ lại seller_id (lấy người đầu tiên hoặc phổ biến nhất)
    # Để đơn giản, chúng ta lấy seller_id đầu tiên. 
    # Trong thực tế, các đơn hàng hỗn hợp (nhiều người bán) hiếm gặp hoặc được xử lý như các đơn hàng phụ.
    # Cấu trúc dữ liệu cho thấy một order_id có thể có nhiều items.
    
    order_items_agg = items_products.groupby('order_id').agg({
        'price': 'sum',
        'shipping_charges': 'sum',
        'product_weight_g': 'sum',
        'product_length_cm': 'max',
        'product_height_cm': 'max',
        'product_width_cm': 'max',
        'product_category_name': 'first',
        'seller_id': 'first', # Giữ lại seller_id
        'product_id': 'count' # Số lượng items
    }).rename(columns={'product_id': 'num_items'}).reset_index()
    
    # Tính phí vận chuyển trên mỗi đơn vị trọng lượng
    # Tránh chia cho 0
    order_items_agg['shipping_per_weight'] = order_items_agg['shipping_charges'] / (order_items_agg['product_weight_g'] + 1.0)
    
    # Gộp Orders với Items đã tổng hợp
    df = orders.merge(order_items_agg, on='order_id', how='inner')
    
    # Gộp với Customers
    df = df.merge(customers, on='customer_id', how='left')
    
    # Gộp với Payments (Tổng hợp trước)
    # Chúng ta muốn payment_type và payment_value
    payments_agg = payments.groupby('order_id').agg({
        'payment_type': 'first', # Thường là cùng loại
        'payment_value': 'sum',
        'payment_installments': 'max'
    }).reset_index()
    
    df = df.merge(payments_agg, on='order_id', how='left')
    
    print(f"Kích thước dữ liệu sau khi gộp: {df.shape}")
    
    # 2. Chuyển đổi ngày tháng
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_timestamp', 'order_estimated_delivery_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    # 3. Tính toán Mục tiêu: delivery_days
    # Sử dụng order_delivered_timestamp - order_purchase_timestamp
    df = df.dropna(subset=['order_delivered_timestamp', 'order_purchase_timestamp'])
    df['delivery_days'] = (df['order_delivered_timestamp'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400
    
    # 4. Lọc dữ liệu không hợp lệ
    # Loại bỏ ngày giao hàng âm hoặc bằng 0 (không hợp lệ)
    df = df[df['delivery_days'] > 0]
    
    # Loại bỏ trùng lặp
    df = df.drop_duplicates()
    
    print(f"Kích thước dữ liệu sau khi lọc cơ bản: {df.shape}")
    
    # 5. Tạo đặc trưng (Feature Engineering)
    # Các đặc trưng về ngày tháng
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_dow'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['is_weekend'] = df['purchase_dow'].isin([5, 6]).astype(int)
    
    # Thời gian ước tính (Ước tính của hệ thống logistics tại thời điểm mua hàng)
    df['estimated_duration'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400
    
    # Thể tích
    df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    
    # ===== CÁC ĐẶC TRƯNG MỚI ĐỂ CẢI THIỆN MÔ HÌNH =====
    df = features.create_features(df)
    
    # Điền giá trị thiếu cho các đặc trưng số bằng 0 hoặc trung vị
    num_cols = ['product_weight_g', 'product_volume_cm3', 'product_length_cm', 'product_height_cm', 
                'product_width_cm', 'payment_value', 'payment_installments', 'shipping_per_weight',
                'distance_estimate_km', 'seller_avg_delivery_days', 'seller_order_count', 'seller_delivery_std',
                'category_avg_delivery_days', 'category_delivery_std', 'customer_order_count']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        
    # Điền giá trị thiếu cho các đặc trưng phân loại
    df['product_category_name'] = df['product_category_name'].fillna('unknown')
    df['payment_type'] = df['payment_type'].fillna('unknown')
    df['seller_id'] = df['seller_id'].fillna('unknown')
    
    # 6. Xử lý ngoại lai & Tạo nhãn
    # Định nghĩa ngưỡng
    THRESHOLD_EXTREME = 60 # ngày
    
    # Tạo cờ is_extreme
    df['is_extreme'] = (df['delivery_days'] > THRESHOLD_EXTREME).astype(int)
    
    # Tạo nhãn phân loại
    # 0: Bình thường, 1: Trễ (nhưng không cực kỳ), 2: Cực kỳ trễ
    
    # Tính độ trễ so với ước tính
    df['delay_vs_estimated'] = (df['order_delivered_timestamp'] - df['order_estimated_delivery_date']).dt.total_seconds() / 86400
    
    def classify_delivery(row):
        if row['delivery_days'] > 60:
            return 2 # Cực kỳ trễ
        elif row['order_delivered_timestamp'] > row['order_estimated_delivery_date']:
            return 1 # Trễ
        else:
            return 0 # Bình thường
            
    df['delivery_class'] = df.apply(classify_delivery, axis=1)
    
    # Áp dụng giới hạn cho mục tiêu hồi quy (Giai đoạn 2)
    df['delivery_days_capped'] = df['delivery_days'].clip(upper=60)
    
    print("Hoàn thành tiền xử lý.")
    return df

if __name__ == "__main__":
    orders, items, products, customers, payments = load_data()
    df_clean = preprocess_data(orders, items, products, customers, payments)
    
    # Lưu vào đĩa
    output_path = os.path.join(DATA_DIR, 'processed_data.csv')
    df_clean.to_csv(output_path, index=False)
    print(f"Đã lưu dữ liệu đã xử lý vào {output_path}")
