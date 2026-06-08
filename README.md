# EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm

Bộ file Python đã được xây dựng lại theo hướng **ML demo + LCA/ISO decision-support prototype**.

## File chính

- `carbon_utils.py`: xử lý dữ liệu, feature engineering, mô hình, LCA bottom-up, uncertainty, OOD check.
- `carbon_eda.py`: trực quan hóa EDA, log histogram, trend theo năm, missing values, lifecycle stacked bar.
- `train_advanced_models.py`: huấn luyện nhiều mô hình phân loại/hồi quy, time-based split, metric, lưu model.
- `app.py`: giao diện Streamlit EcoPredict Carbon.
- `requirements.txt`: thư viện cần cài.
- `carbon_catalogue.csv`: dữ liệu đầu vào.

## Chạy trên Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python carbon_eda.py
python train_advanced_models.py
python -m streamlit run app.py
```

## Điểm cải tiến chính

- Giữ nguyên tiêu đề giao diện: **EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm**.
- Sửa biểu đồ PCF sang log-scale để xử lý phân phối lệch phải.
- Thêm time-based split thay vì random split để mô phỏng dự báo tương lai.
- Thêm uncertainty interval P10–P90 dựa trên phần dư test.
- Thêm OOD check để cảnh báo khi input ngoài miền dữ liệu huấn luyện.
- Thêm LCA bottom-up inventory editor: PCF = Σ(activity × emission factor).
- Thêm Hybrid PCF = kết hợp ML và LCA bottom-up khi có inventory.
- Thêm scenario forecasting 2026–2035.
- Thêm trang LCA/ISO nâng cao: Goal & Scope, Data Quality Rating, audit trail.
- Dùng stacked bar cho lifecycle thay vì donut.

## Lưu ý học thuật

Đây là prototype hỗ trợ ra quyết định, không phải chứng nhận ISO/LCA chính thức. Muốn thành hệ thống LCA/ISO đầy đủ cần cơ sở emission factors được kiểm định, audit trail đầy đủ, reviewer độc lập và kiểm chứng chuyên gia.
