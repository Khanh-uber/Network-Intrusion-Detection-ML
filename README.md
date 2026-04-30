# Network Intrusion Detection System (NIDS) - CIC-IDS2017

Dự án sử dụng Machine Learning để phát hiện xâm nhập mạng dựa trên bộ dữ liệu CIC-IDS2017. 

## 👥 Thành viên thực hiện
- **Nguyễn Minh Khánh:** Thiết lập môi trường, Tiền xử lý dữ liệu (Preprocessing), Phân tích dữ liệu (EDA), Xây dựng hệ thống demo (Main).
- **Trần Minh Khôi:** Huấn luyện mô hình (Random Forest, SVM, KNN), Đánh giá hiệu suất (Metrics).

## 📂 Cấu trúc thư mục
Network-Intrusion-Detection-ML/
├── notebooks/
│   ├── 1_EDA_Preprocessing.ipynb     
│   └── 2_Model_Comparison.ipynb      
├── src/
│   ├── init.py
│   ├── preprocessing.py              
│   └── models/                       
│       ├── init.py
│       ├── logistic_regression.py
│       ├── svm.py
│       ├── naive_bayes.py
│       ├── knn.py
│       └── random_forest.py          
├── models_saved/                     
│   └── best_rf_model.pkl
└── main.py                           

## 🚀 Cài đặt môi trường
1. Tạo môi trường ảo:
   ```bash
   python -m venv env
   ```
2. Kích hoạt môi trường:
   - Windows: `env\Scripts\activate`
   - Mac/Linux: `source env/bin/activate`
3. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Cách chạy dự án
(Sẽ cập nhật sau khi hoàn thiện các module)
