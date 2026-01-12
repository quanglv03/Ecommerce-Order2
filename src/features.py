import pandas as pd
import numpy as np

def create_features(df):
    """
    Tạo các đặc trưng mới để cải thiện mô hình.
    """
    print("Đang thêm các đặc trưng mới để cải thiện mô hình...")
    
    # 1. Ước tính khoảng cách địa lý (Khách hàng <-> Người bán)
    # Vì chúng ta không có mã bưu chính người bán trong tất cả các bộ dữ liệu, chúng ta sẽ dùng proxy:
    # Tính toán phương sai của mã bưu chính khách hàng như một thước đo phân tán địa lý
    # Hiện tại, sử dụng cách tiếp cận đơn giản: giả định khoảng cách dựa trên phân phối mã bưu chính khách hàng
    
    # Chuyển đổi mã bưu chính khách hàng sang dạng số và chuẩn hóa
    df['customer_zip_numeric'] = pd.to_numeric(df['customer_zip_code_prefix'], errors='coerce')
    
    # Tạo ước tính khoảng cách: độ lệch so với mã bưu chính trung vị * hệ số tỷ lệ
    median_zip = df['customer_zip_numeric'].median()
    df['distance_estimate_km'] = (df['customer_zip_numeric'] - median_zip).abs() / 10
    
    # Giới hạn khoảng cách tối đa hợp lý (ví dụ: 500km)
    df['distance_estimate_km'] = df['distance_estimate_km'].clip(upper=500)
    
    # Dọn dẹp cột tạm thời
    df = df.drop(columns=['customer_zip_numeric'], errors='ignore')
    
    
    # 2. Lịch sử hiệu suất người bán
    # Tính thời gian giao hàng trung bình cho mỗi người bán (sử dụng tất cả dữ liệu làm proxy cho hiệu suất lịch sử)
    # Lưu ý: Trong thực tế (production), chỉ nên sử dụng dữ liệu lịch sử trước ngày đặt hàng
    seller_stats = df.groupby('seller_id')['delivery_days'].agg(['mean', 'count', 'std']).reset_index()
    seller_stats.columns = ['seller_id', 'seller_avg_delivery_days', 'seller_order_count', 'seller_delivery_std']
    
    # Điền 0 cho std bị NaN (người bán chỉ có 1 đơn hàng)
    seller_stats['seller_delivery_std'] = seller_stats['seller_delivery_std'].fillna(0)
    
    # Gộp lại vào dataframe chính
    df = df.merge(seller_stats, on='seller_id', how='left')
    
    # 3. Lịch sử hiệu suất danh mục sản phẩm
    # Tính thời gian giao hàng trung bình cho mỗi danh mục
    category_stats = df.groupby('product_category_name')['delivery_days'].agg(['mean', 'std']).reset_index()
    category_stats.columns = ['product_category_name', 'category_avg_delivery_days', 'category_delivery_std']
    category_stats['category_delivery_std'] = category_stats['category_delivery_std'].fillna(0)
    
    # Gộp lại
    df = df.merge(category_stats, on='product_category_name', how='left')
    
    # 4. Lịch sử khách hàng (nếu có khách hàng mua lại)
    customer_stats = df.groupby('customer_id').size().reset_index(name='customer_order_count')
    df = df.merge(customer_stats, on='customer_id', how='left')
    
    # 5. Các đặc trưng thời gian nâng cao
    # Hiệu ứng cuối tháng (đơn hàng gần cuối tháng có thể bị trễ)
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['is_month_end'] = (df['purchase_day'] > 25).astype(int)
    
    # Hiệu ứng mùa lễ (Tháng 11-12 thường bận rộn hơn)
    df['is_holiday_season'] = df['purchase_month'].isin([11, 12]).astype(int)
    
    # Quý trong năm
    df['purchase_quarter'] = df['order_purchase_timestamp'].dt.quarter
    
    print(f"Đã thêm các đặc trưng mới. Kích thước hiện tại: {df.shape}")
    
    return df
