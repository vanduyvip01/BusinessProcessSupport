import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. Tải dữ liệu và kết hợp ---
# Giả định các file train.csv và store.csv đã có sẵn trong cùng thư mục.
# Nếu không, code sẽ tạo dữ liệu giả để minh họa.
try:
    sales_train_df = pd.read_csv('train.csv', low_memory=False)
    store_info_df = pd.read_csv('store.csv', low_memory=False)
    print("Dữ liệu gốc đã được tải thành công.")
except FileNotFoundError:
    print("Không tìm thấy file dữ liệu. Tạo dữ liệu giả để minh họa.")
    # Dữ liệu giả để minh họa nếu không có file gốc
    sales_data = {
        'Store': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'DayOfWeek': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
        'Date': pd.to_datetime(pd.date_range(start='2015-07-01', periods=10).append(pd.date_range(start='2015-07-01', periods=10))),
        'Sales': [5263, 6064, 8314, 13995, 4822, 5651, 15344, 8492, 8565, 7185, 6000, 6500, 8500, 14000, 5000, 5800, 15500, 8600, 8700, 7300],
        'Customers': [555, 625, 821, 1498, 559, 589, 1414, 833, 687, 681, 600, 650, 830, 1500, 560, 590, 1420, 840, 690, 685],
        'Open': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'Promo': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'StateHoliday': ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
        'SchoolHoliday': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    sales_train_df = pd.DataFrame(sales_data)
    store_info_data = {
        'Store': [1, 2],
        'StoreType': ['c', 'a'],
        'Assortment': ['a', 'a'],
        'CompetitionDistance': [1270.0, 570.0],
        'CompetitionOpenSinceMonth': [9.0, 11.0],
        'CompetitionOpenSinceYear': [2008.0, 2007.0],
        'Promo2': [0, 1],
        'Promo2SinceWeek': [np.nan, 13.0],
        'Promo2SinceYear': [np.nan, 2010.0],
        'PromoInterval': [np.nan, 'Jan,Apr,Jul,Oct']
    }
    store_info_df = pd.DataFrame(store_info_data)

sales_train_all_df = pd.merge(sales_train_df, store_info_df, on='Store', how='inner')
print("Kết hợp dữ liệu thành công.")

# --- 2. Làm sạch dữ liệu và tiền xử lý ---
print("Bắt đầu tiền xử lý dữ liệu...")
# Xử lý các giá trị NaN
sales_train_all_df['CompetitionDistance'].fillna(sales_train_all_df['CompetitionDistance'].median(), inplace=True)
for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']:
    sales_train_all_df[col].fillna(0, inplace=True)

# Chuyển đổi cột 'Date' sang định dạng datetime
sales_train_all_df['Date'] = pd.to_datetime(sales_train_all_df['Date'])

# Lọc các bản ghi khi cửa hàng đóng cửa hoặc không có doanh số
sales_train_all_df = sales_train_all_df[(sales_train_all_df['Open'] == 1) & (sales_train_all_df['Sales'] > 0)]
sales_train_all_df.drop(['Open'], axis=1, inplace=True)

# --- 3. Tạo đặc trưng ---
print("Bắt đầu tạo đặc trưng...")
# Tạo các đặc trưng thời gian
sales_train_all_df['Year'] = sales_train_all_df['Date'].dt.year
sales_train_all_df['Month'] = sales_train_all_df['Date'].dt.month
sales_train_all_df['Day'] = sales_train_all_df['Date'].dt.day
sales_train_all_df['WeekOfYear'] = sales_train_all_df['Date'].dt.isocalendar().week.astype(int)

# Mã hóa các biến phân loại bằng One-Hot Encoding
categorical_features = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']
sales_train_all_df = pd.get_dummies(sales_train_all_df, columns=categorical_features, drop_first=True)

# Chuẩn hóa dữ liệu cho LSTM
features = [col for col in sales_train_all_df.columns if col not in ['Date', 'Sales', 'Customers']]
target = 'Sales'

# Xử lý các cột bị thiếu do get_dummies với dữ liệu giả
for col in features:
    if col not in sales_train_all_df.columns:
        sales_train_all_df[col] = 0

data_to_scale = sales_train_all_df[features + [target]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_to_scale)

# Tạo chuỗi dữ liệu cho LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

SEQUENCE_LENGTH = 7  # Sử dụng dữ liệu 7 ngày trước để dự đoán ngày tiếp theo
X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

# --- 4. Phát triển mô hình LSTM ---
print("Bắt đầu phát triển và huấn luyện mô hình LSTM...")
# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Đánh giá trên tập kiểm tra
predictions = model.predict(X_test)

# Chuyển đổi ngược các giá trị dự đoán và thực tế về thang ban đầu
y_test_reshaped = np.zeros((y_test.shape[0], scaled_data.shape[1]))
y_test_reshaped[:, -1] = y_test
y_test_original = scaler.inverse_transform(y_test_reshaped)[:, -1]

predictions_reshaped = np.zeros((predictions.shape[0], scaled_data.shape[1]))
predictions_reshaped[:, -1] = predictions.flatten()
predictions_original = scaler.inverse_transform(predictions_reshaped)[:, -1]

# Tính toán RMSE
rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
print(f"\nRMSE trên tập kiểm tra: {rmse}")

# Vẽ biểu đồ kết quả
plt.figure(figsize=(15, 6))
plt.plot(y_test_original[:100], label='Doanh số thực tế')
plt.plot(predictions_original[:100], label='Dự đoán của LSTM')
plt.title('Dự đoán doanh số bằng mô hình LSTM')
plt.xlabel('Mẫu')
plt.ylabel('Doanh số')
plt.legend()
plt.show()

