import pandas as pd
import numpy as np
import glob

# Danh sách 18 đặc trưng (Đã thay Protocol bằng Dst Port)
SELECTED_FEATURES = [
    'Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Mean', 'Flow Byts/s', 'Flow Pkts/s',
    'Pkt Len Mean', 'Pkt Len Std', 'SYN Flag Cnt',
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Count', 'URG Flag Count'
]

def load_and_preprocess(data):

    # 1. Lấy danh sách file và gộp dữ liệu
    all_files = glob.glob(data + "/*.csv")
    df_list = []
    
    for filename in all_files:
        print(f"Đang đọc file: {filename}")
        df = pd.read_csv(filename)
        
        # 2. Làm sạch tên cột ngay khi đọc (Tránh lỗi khoảng trắng)
        df.columns = df.columns.str.strip()
        
        # 3. Tối ưu bộ nhớ (Downcast) - Bỏ qua cột nhãn
        for col in df.columns:
            if col != 'Label':
                if df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
        
        df_list.append(df)
    
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

def clean_data(df):

    # 1. Xử lý giá trị vô hạn (Inf) thành NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Xử lý NaN bằng Median thay vì drop
    # Chỉ tính median cho các cột số
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 3. Xóa các dòng trùng lặp và các cột có phương sai bằng 0 (Zero variance)
    df.drop_duplicates(inplace=True)
    
    # Xóa cột có toàn bộ giá trị giống nhau (không có giá trị dự báo)
    non_zero_var_cols = [col for col in df.columns if df[col].nunique() > 1 or col == 'Label']
    df = df[non_zero_var_cols]
    
    return df
