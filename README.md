# EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm

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

## Tổng quan hệ thống

EcoPredict Carbon là hệ thống hỗ trợ dự báo và phân tích phát thải carbon của sản phẩm theo hướng PCF/LCA. Hệ thống kết hợp dữ liệu Carbon Catalogue, OpenPCF và Open CEDA để ước lượng Product Carbon Footprint (PCF), phân loại mức phát thải, so sánh với nhóm ngành và mô phỏng các kịch bản giảm phát thải trong tương lai.

Trọng tâm của hệ thống không chỉ là huấn luyện mô hình học máy, mà là xây dựng một quy trình hỗ trợ ra quyết định gồm: dữ liệu đầu vào, tính toán PCF theo tư duy LCA, mô hình Machine Learning, phân tích độ tin cậy, benchmark ngành và giao diện trực quan cho người dùng.

### Mục tiêu chính

* Dự báo lượng phát thải carbon của sản phẩm dưới dạng kg CO₂e.
* Phân loại mức phát thải thành Low / Medium / High.
* So sánh PCF dự báo với trung vị ngành và dữ liệu tham chiếu.
* Hỗ trợ nhập kiểm kê vòng đời cơ bản như vật liệu, năng lượng, vận chuyển, bao bì và cuối vòng đời.
* Mô phỏng kịch bản tương lai đến năm 2050 dựa trên các giả định như tăng điện tái tạo, giảm vật liệu và cải thiện vận chuyển.
* Giải thích các yếu tố ảnh hưởng chính đến kết quả dự báo.
* Hiển thị kết quả trên dashboard Streamlit theo hướng dễ hiểu cho người dùng doanh nghiệp.

### Cách tiếp cận

Hệ thống được định hướng theo mô hình kết hợp giữa PCF, LCA và Machine Learning:

* **PCF** là lõi của bài toán, tập trung vào dự báo dấu chân carbon của sản phẩm.
* **LCA** là nền tảng phương pháp luận, giúp tổ chức dữ liệu theo vòng đời sản phẩm.
* **Machine Learning** hỗ trợ dự báo, hiệu chỉnh sai số, phát hiện bất thường, benchmark ngành và giải thích yếu tố ảnh hưởng.
* **ISO/EPD** được xem là định hướng phát triển minh bạch trong tương lai, chưa phải chứng nhận chính thức.

Công thức nền được sử dụng trong phần kiểm kê vòng đời:

```text
PCF = Σ(activity data × emission factor)
```

### Dữ liệu sử dụng

* `carbon_catalogue.csv`: dữ liệu PCF lịch sử ở cấp sản phẩm.
* `OpenPCF`: bổ sung dữ liệu PCF theo sản phẩm, quốc gia và đặc trưng hoạt động.
* `Open CEDA 2024/2025`: bổ sung hệ số phát thải theo ngành/quốc gia.
* Dữ liệu người dùng nhập trên web: năm kịch bản, khối lượng, quốc gia/khu vực, nhóm ngành, ngành sản phẩm, chuẩn PCF và thông tin kiểm kê vòng đời.

### Thành phần chính của hệ thống

* **Trang Dự báo:** nhập thông tin sản phẩm, dự báo PCF, phân loại phát thải và so sánh với ngành.
* **Trang Dữ liệu:** trực quan hóa phân phối PCF, dữ liệu theo ngành, quốc gia, năm và vòng đời.
* **Trang LCA/ISO:** nhập kiểm kê vòng đời cơ bản và xem định hướng chất lượng dữ liệu.
* **Trang Đánh giá mô hình:** hiển thị kết quả huấn luyện, metric, biểu đồ đánh giá và yếu tố ảnh hưởng.
* **Trang Hướng dẫn:** hướng dẫn người dùng sử dụng hệ thống.

### Ứng dụng Machine Learning

* Sử dụng cả mô hình phân loại và hồi quy.
* Dùng time-based split thay vì chia ngẫu nhiên để phù hợp hơn với bài toán dự báo.
* Bổ sung uncertainty interval để kết quả không chỉ là một con số tuyệt đối.
* Thêm OOD check để đánh giá mềm mức độ tin cậy khi đầu vào khác biệt với dữ liệu đã học.
* Sử dụng permutation importance để giải thích yếu tố ảnh hưởng đến dự báo.

### Vai trò của hệ thống

EcoPredict Carbon được thiết kế như một hệ thống hỗ trợ ra quyết định, giúp người dùng đánh giá sơ bộ phát thải carbon của sản phẩm, so sánh với ngành và thử các kịch bản giảm phát thải. Kết quả của hệ thống mang tính ước lượng tham khảo, không thay thế báo cáo LCA/ISO hoặc chứng nhận EPD chính thức.


## Lưu ý học thuật

Đây là prototype hỗ trợ ra quyết định, không phải chứng nhận ISO/LCA chính thức. Muốn thành hệ thống LCA/ISO đầy đủ cần cơ sở emission factors được kiểm định, audit trail đầy đủ, reviewer độc lập và kiểm chứng chuyên gia.
