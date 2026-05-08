import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Danh sách 18 đặc trưng (Đã thay Protocol bằng Dst Port)
SELECTED_FEATURES = [
    'Protocol',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Fwd Packets Length Total',
    'Bwd Packets Length Total',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Mean',
    'Packet Length Std',
    'SYN Flag Count',
    'ACK Flag Count',
    'FIN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'URG Flag Count'
]

def load_and_preprocess(data):

    # 1. Lấy danh sách file và gộp dữ liệu
    all_files = glob.glob(data + "/*.parquet")
    df_list = []
    
    for filename in all_files:
        print(f"Đang đọc file: {filename}")
        df = pd.read_parquet(filename)
        
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

    combined_df = combined_df.sample(100000, random_state=42)

    

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



def prepare_train_test(df):


    label_counts = df['Label'].value_counts()

    valid_labels = label_counts[label_counts > 10].index

    df = df[df['Label'].isin(valid_labels)]

    # Chọn feature + label
    X = df[SELECTED_FEATURES]
    y = df['Label']

    # Encode label
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


 

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # SMOTE
    smote = SMOTE(
    random_state=42,
    k_neighbors=1
    )
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # UnderSampling
    under = RandomUnderSampler(random_state=42)
    X_train, y_train = under.fit_resample(X_train, y_train)

    # Save scaler + encoder
    joblib.dump(scaler, 'models_saved/scaler.pkl')
    joblib.dump(label_encoder, 'models_saved/label_encoder.pkl')

    # Convert lại DataFrame
    X_train = pd.DataFrame(X_train, columns=SELECTED_FEATURES)
    X_test = pd.DataFrame(X_test, columns=SELECTED_FEATURES)

    return X_train, X_test, y_train, y_test



