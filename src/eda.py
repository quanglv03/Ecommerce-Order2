import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_data.csv')

def run_eda(file_path=DATA_PATH):
    print("Đang tải dữ liệu cho EDA...")
    df = pd.read_csv(file_path)
    
    print("\n--- Tổng quan dữ liệu ---")
    print(df.info())
    
    print("\n--- Giá trị bị thiếu ---")
    print(df.isnull().sum())
    
    print("\n--- Phân phối ngày giao hàng ---")
    print(df['delivery_days'].describe())
    
    print("\n--- Phân vị ---")
    percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    print(df['delivery_days'].quantile(percentiles))
    
    print("\n--- Phân tích ngoại lai ---")
    print(f"Số lượng > 60 ngày: {len(df[df['delivery_days'] > 60])}")
    print(f"Số lượng > 90 ngày: {len(df[df['delivery_days'] > 90])}")
    
    print("\n--- Phân phối lớp ---")
    print(df['delivery_class'].value_counts(normalize=True))
    print("0: Bình thường, 1: Trễ, 2: Cực kỳ trễ")
    
    print("\n--- Tương quan với mục tiêu ---")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()['delivery_days'].sort_values(ascending=False)
    print(corr.head(10))
    print(corr.tail(5))
    
    # ===== TRỰC QUAN HÓA DỮ LIỆU =====
    print("\n--- Tạo biểu đồ trực quan ---")
    
    # Cấu hình style cho đẹp hơn
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Phân phối ngày giao hàng
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0, 0].hist(df['delivery_days'], bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Phân phối ngày giao hàng', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Số ngày giao hàng')
    axes[0, 0].set_ylabel('Tần suất')
    axes[0, 0].axvline(df['delivery_days'].mean(), color='red', linestyle='--', label=f'Mean: {df["delivery_days"].mean():.2f}')
    axes[0, 0].axvline(df['delivery_days'].median(), color='green', linestyle='--', label=f'Median: {df["delivery_days"].median():.2f}')
    axes[0, 0].legend()
    
    # Boxplot
    axes[0, 1].boxplot(df['delivery_days'], vert=True)
    axes[0, 1].set_title('Boxplot ngày giao hàng', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Số ngày')
    
    # Phân phối theo lớp
    class_counts = df['delivery_class'].value_counts().sort_index()
    axes[1, 0].bar(['Bình thường', 'Trễ', 'Cực trễ'], class_counts.values, color=['green', 'orange', 'red'])
    axes[1, 0].set_title('Phân phối theo lớp giao hàng', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Loại giao hàng')
    axes[1, 0].set_ylabel('Số lượng')
    for i, v in enumerate(class_counts.values):
        axes[1, 0].text(i, v + 500, str(v), ha='center', fontweight='bold')
    
    # Phân phối theo tháng
    df_copy = df.copy()
    if 'purchase_month' in df_copy.columns:
        month_counts = df_copy['purchase_month'].value_counts().sort_index()
        axes[1, 1].plot(month_counts.index, month_counts.values, marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_title('Số đơn hàng theo tháng', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Tháng')
        axes[1, 1].set_ylabel('Số đơn hàng')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Không có dữ liệu tháng', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('eda_distribution_plots.png', dpi=300, bbox_inches='tight')
    print("Đã lưu: eda_distribution_plots.png")
    plt.show()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(14, 10))
    
    # Chọn các cột số quan trọng
    important_cols = ['delivery_days', 'price', 'shipping_charges', 'product_weight_g', 
                      'num_items', 'payment_value', 'estimated_duration', 'product_volume_cm3',
                      'distance_estimate_km', 'seller_avg_delivery_days', 'category_avg_delivery_days']
    
    # Lọc các cột tồn tại
    available_cols = [col for col in important_cols if col in numeric_df.columns]
    
    if len(available_cols) > 1:
        corr_matrix = numeric_df[available_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Ma trận tương quan các đặc trưng quan trọng', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Đã lưu: eda_correlation_heatmap.png")
        plt.show()
    
    # 3. Phân tích theo danh mục sản phẩm
    if 'product_category_name' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Top 15 danh mục có nhiều đơn hàng nhất
        top_categories = df['product_category_name'].value_counts().head(15)
        
        plt.subplot(1, 2, 1)
        top_categories.plot(kind='barh', color='steelblue')
        plt.title('Top 15 danh mục sản phẩm', fontsize=14, fontweight='bold')
        plt.xlabel('Số đơn hàng')
        plt.ylabel('Danh mục')
        plt.gca().invert_yaxis()
        
        # Thời gian giao hàng trung bình theo danh mục
        plt.subplot(1, 2, 2)
        category_delivery = df.groupby('product_category_name')['delivery_days'].mean().sort_values(ascending=False).head(15)
        category_delivery.plot(kind='barh', color='coral')
        plt.title('Top 15 danh mục có thời gian giao lâu nhất', fontsize=14, fontweight='bold')
        plt.xlabel('Ngày giao hàng trung bình')
        plt.ylabel('Danh mục')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('eda_category_analysis.png', dpi=300, bbox_inches='tight')
        print("Đã lưu: eda_category_analysis.png")
        plt.show()
    
    # 4. Phân tích theo khoảng cách và trọng lượng
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'distance_estimate_km' in df.columns:
        axes[0].scatter(df['distance_estimate_km'], df['delivery_days'], alpha=0.3, s=10)
        axes[0].set_title('Thời gian giao hàng vs Khoảng cách', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Khoảng cách ước tính (km)')
        axes[0].set_ylabel('Ngày giao hàng')
        axes[0].grid(True, alpha=0.3)
    
    if 'product_weight_g' in df.columns:
        # Lọc outliers để nhìn rõ hơn
        weight_filtered = df[df['product_weight_g'] < df['product_weight_g'].quantile(0.95)]
        axes[1].scatter(weight_filtered['product_weight_g'], weight_filtered['delivery_days'], alpha=0.3, s=10, color='green')
        axes[1].set_title('Thời gian giao hàng vs Trọng lượng', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Trọng lượng sản phẩm (g)')
        axes[1].set_ylabel('Ngày giao hàng')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_scatter_plots.png', dpi=300, bbox_inches='tight')
    print("Đã lưu: eda_scatter_plots.png")
    plt.show()
    
    print("\nHoàn thành EDA với trực quan hóa!")
    print("Đã tạo 4 biểu đồ:")
    print("  1. eda_distribution_plots.png")
    print("  2. eda_correlation_heatmap.png")
    print("  3. eda_category_analysis.png")
    print("  4. eda_scatter_plots.png")

if __name__ == "__main__":
    run_eda()
