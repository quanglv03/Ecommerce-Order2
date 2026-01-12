# Dự đoán Giao hàng Logistics (Logistics Delivery Prediction)

Dự án này xây dựng một hệ thống học máy để dự đoán thời gian giao hàng và đánh giá hiệu suất giao hàng trong lĩnh vực thương mại điện tử. Hệ thống sử dụng mô hình hai giai đoạn: phân loại (Classifier) để dự đoán khả năng giao hàng trễ/sớm, và hồi quy (Regressor) để ước tính số ngày giao hàng cụ thể.

## Cấu trúc Dự án

Dự án được tổ chức trong thư mục `Ecommer2` với các thành phần chính sau:

- **`main.py`**: File chính để chạy toàn bộ quy trình (pipeline) từ tiền xử lý, EDA, huấn luyện mô hình đến đánh giá.
- **`preprocessing.py`**: Module tiền xử lý dữ liệu, bao gồm làm sạch, gộp bảng và chuẩn bị dữ liệu.
- **`features.py`**: Module chứa logic tạo đặc trưng (feature engineering), bao gồm tính toán khoảng cách, thống kê lịch sử người bán/khách hàng, v.v.
- **`eda.py`**: Module Phân tích Dữ liệu Khám phá (Exploratory Data Analysis) để trực quan hóa và hiểu dữ liệu.
- **`model_classifier_catboost.py`**: Huấn luyện mô hình phân loại CatBoost (Giai đoạn 1).
- **`model_regressor_catboost.py`**: Huấn luyện mô hình hồi quy CatBoost (Giai đoạn 2).
- **`model_regressor_xgboost.py`**: Huấn luyện mô hình hồi quy XGBoost (Giai đoạn 2).
- **`evaluation.py`**: Đánh giá và so sánh hiệu suất của các mô hình.
- **`data/`**: Thư mục chứa dữ liệu thô (csv) và dữ liệu đã xử lý.
- **`models/`**: Thư mục lưu trữ các mô hình đã được huấn luyện (.pkl).

## Yêu cầu Hệ thống

- Python 3.8 trở lên
- Các thư viện Python cần thiết:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - catboost

## Hướng dẫn Cài đặt

1.  **Clone hoặc tải dự án về máy.**

2.  **Tạo môi trường ảo (khuyến nghị):**

    ```bash
    python -m venv .venv
    ```

3.  **Kích hoạt môi trường ảo:**

    - Trên Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - Trên macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

4.  **Cài đặt các thư viện phụ thuộc:**
    Nếu chưa có file `requirements.txt`, bạn có thể cài đặt thủ công các thư viện chính:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
    ```

## Hướng dẫn Chạy Dự án

Để chạy toàn bộ quy trình dự đoán, hãy thực thi file `main.py` từ thư mục gốc của dự án:

```bash
python Ecommer2/main.py
```

Quy trình sẽ thực hiện tuần tự các bước:

1.  **Tiền xử lý**: Tải dữ liệu thô, làm sạch, tạo đặc trưng và lưu vào `data/processed_data.csv`.
2.  **EDA**: Hiển thị các biểu đồ phân tích dữ liệu cơ bản.
3.  **Huấn luyện Phân loại**: Huấn luyện mô hình CatBoost để phân loại đơn hàng.
4.  **Huấn luyện Hồi quy**: Huấn luyện các mô hình CatBoost và XGBoost để dự đoán thời gian giao hàng.
5.  **Đánh giá**: So sánh kết quả và hiển thị các chỉ số đánh giá (Accuracy, F1, MAE, RMSE, R2).

## Ghi chú

- Đảm bảo dữ liệu thô (các file csv như `df_Orders.csv`, `df_OrderItems.csv`, v.v.) đã có sẵn trong thư mục `Ecommer2/data/`.
- Mô hình đã huấn luyện sẽ được lưu vào thư mục `Ecommer2/models/` để tái sử dụng.
