import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def parse_level(x):
    """
    Xử lý giá trị Flight_level, tính trung bình nếu có dấu chia (/),
    nếu lỗi thì trả về 0.0.
    """
    try:
        if isinstance(x, str) and "/" in x:
            vals = [float(v.strip()) for v in x.split("/") if v.strip()]
            return sum(vals) / len(vals) if vals else 0.0
        return float(x)
    except:
        return 0.0

print("1️⃣ Load và xử lý dữ liệu...")
df = pd.read_csv("train_file.csv")
original_rows = len(df)
if 'STT' not in df.columns:
    df.insert(0, 'STT', df.index + 1)
print(f"   Đã tải: {original_rows} dòng.")

df['Flight_level'] = df['Flight_level'].apply(parse_level)

def to_dt(date_col, time_col):
    """Chuyển đổi cột ngày và giờ thành đối tượng datetime. Dùng MM/DD/YYYY."""
    s = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
    return pd.to_datetime(s, dayfirst=False, utc=False, errors="coerce") 

df["takeoff_dt"] = to_dt("Take_off_date", "Take_off_time")
df["landing_dt"] = to_dt("Landing_date", "Landing_time")

mask_invalid = df["takeoff_dt"].isna() | df["landing_dt"].isna()
df_invalid = df[mask_invalid].copy()
df_cleaned = df[~mask_invalid].copy()

cleaned_rows = len(df_cleaned)
dropped_rows = original_rows - cleaned_rows
print(f"   Loại bỏ: {dropped_rows} dòng do thiếu/sai giờ bay.")
print(f"   Dữ liệu sạch còn lại: {cleaned_rows} dòng.")
df = df_cleaned

if dropped_rows > 0:
    print("--- ❌ BÁO CÁO DÒNG BỊ LOẠI BỎ (Do lỗi định dạng thời gian) ❌ ---")
    print(f"Các dòng bị loại bỏ có STT: {df_invalid['STT'].tolist()}")
    print("\nChi tiết 5 dòng đầu tiên bị lỗi (Kiểm tra cột *date* và *time* trong file gốc):")
    print(df_invalid[['STT', 'Take_off_date', 'Take_off_time', 'Landing_date', 'Landing_time']].head(5).to_markdown(index=False))
    print("------------------------------------------------------------------")

mask = (df["landing_dt"] < df["takeoff_dt"]) & ((df["takeoff_dt"] - df["landing_dt"]) < pd.Timedelta(hours=4))
df.loc[mask, "landing_dt"] += pd.Timedelta(days=1)
df["duration_min"] = (df["landing_dt"] - df["takeoff_dt"]).dt.total_seconds() / 60
df["duration_hr"] = df["duration_min"] / 60

print("2️⃣ Tính toán thời gian bay trung bình...")
df["route"] = df["Departure_airport"] + "-" + df["Arrival_airport"]
route_stats = df.groupby("route")["duration_hr"].agg(["mean", "count"]).reset_index()
route_mean = dict(zip(route_stats["route"], route_stats["mean"]))
route_count = dict(zip(route_stats["route"], route_stats["count"]))
global_mean = route_stats["mean"].median() 

def get_mean(row):
    route = row["route"]
    mean_val = route_mean.get(route, global_mean)
    count_val = route_count.get(route, 0)
    return mean_val if count_val >= 5 else global_mean

df["mean_duration_hr"] = df.apply(get_mean, axis=1)
df["residual_min"] = (df["duration_hr"] - df["mean_duration_hr"]) * 60

print("3️⃣ Chuẩn bị features...")
df["takeoff_hour"] = df["takeoff_dt"].dt.hour
df["takeoff_minute"] = df["takeoff_dt"].dt.minute
df["takeoff_dayofweek"] = df["takeoff_dt"].dt.dayofweek

df_ohe = pd.get_dummies(df[["Departure_airport", "Arrival_airport"]], prefix=["dep", "arr"])
df = pd.concat([df, df_ohe], axis=1)

# Chọn cột features
TIME_FEATURES = ['takeoff_hour', 'takeoff_minute', 'takeoff_dayofweek']
OHE_FEATURES = [col for col in df.columns if col.startswith(('dep_', 'arr_'))]
NUM_FEATURES = TIME_FEATURES + OHE_FEATURES + ['Flight_level']

TARGET = "residual_min"

X_train, y_train = df.loc[:, NUM_FEATURES], df.loc[:, TARGET]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("4️⃣ Huấn luyện mô hình XGBoostRegressor...")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = cross_val_predict(model, X_train, y_train, cv=kf)
mae = mean_absolute_error(y_train, y_pred)
print(f"✅ MAE (phút): {mae:.2f}")
print("5️⃣ Dự đoán thời lượng bay và cập nhật giờ hạ cánh...")


X = df.loc[:, NUM_FEATURES]

duration_min_mean = df["mean_duration_hr"] * 60

takeoff_dt = df["takeoff_dt"]

residual_min = model.predict(X)

predicted_duration_min = duration_min_mean + residual_min
predicted_timedelta = pd.to_timedelta(predicted_duration_min, unit='m')

df["landing_dt_pred"] = takeoff_dt + predicted_timedelta
df["landing_dt"] = df["landing_dt_pred"]
df["duration_min"] = (df["landing_dt"] - df["takeoff_dt"]).dt.total_seconds() / 60
df["duration_hr"] = df["duration_min"] / 60
df["duration_delta_hr"] = df["duration_hr"] - df["mean_duration_hr"]

print(f"Độ lệch trung bình tuyệt đối của thời lượng bay dự kiến so với trung bình (giờ): {df['duration_delta_hr'].abs().mean():.2f}")
outliers = df[df['duration_delta_hr'].abs() > 2].shape[0]
print(f"Số lượng chuyến bay có độ lệch > 2 giờ: {outliers} / {len(df)} ({outliers/len(df)*100:.2f}%)")
print("✅ Đã cập nhật giờ hạ cánh dự kiến chính xác hơn.")


print("6️⃣ Tối ưu hóa lịch trình tại VVNB (Áp dụng luật 2p/0p) cho tất cả các chuyến bay liên quan...")

# CÁC CỘT THUỘC TÍNH GỐC CẦN GIỮ LẠI
ORIGINAL_ATTRIBUTES = ['Aircraft', 'Aircraft_type', 'Flight', 'Aircraft_registration', 
                       'Flight_route', 'Flight_type', 'Flight_purpose', 'Flight_level']
TIME_AND_ROUTE = ['takeoff_dt', 'landing_dt', 'Departure_airport', 'Arrival_airport', 'duration_hr']
ALL_REQUIRED_COLS = TIME_AND_ROUTE + ORIGINAL_ATTRIBUTES

# SỬA LỖI CÚ PHÁP LỌC DỮ LIỆU
df_vvnb_events = df[(df['Departure_airport'] == "VVNB") | (df['Arrival_airport'] == "VVNB")][ALL_REQUIRED_COLS].copy()

# Tách dữ liệu thành 2 phần để xử lý
vvnb_dep = df_vvnb_events[df_vvnb_events["Departure_airport"] == "VVNB"].copy()
vvnb_arr = df_vvnb_events[df_vvnb_events["Arrival_airport"] == "VVNB"].copy()

# CHỌN TẤT CẢ CÁC CỘT CẦN THIẾT KHI TẠO SCHEDULE
DEP_SCHEDULE_COLS = ['takeoff_dt', 'Arrival_airport', 'duration_hr'] + ORIGINAL_ATTRIBUTES + ['Departure_airport']
ARR_SCHEDULE_COLS = ['landing_dt', 'Departure_airport', 'duration_hr'] + ORIGINAL_ATTRIBUTES + ['Arrival_airport']

# Tạo DataFrame cho CẤT CÁNH
schedule_dep = vvnb_dep[DEP_SCHEDULE_COLS].rename(columns={"takeoff_dt": "time", "Arrival_airport": "Target"})
schedule_dep = schedule_dep.assign(type="Takeoff")

# Tạo DataFrame cho HẠ CÁNH
schedule_arr = vvnb_arr[ARR_SCHEDULE_COLS].rename(columns={"landing_dt": "time", "Departure_airport": "Target"})
schedule_arr = schedule_arr.assign(type="Landing")


# Ghép và sắp xếp lịch trình
schedule = pd.concat([schedule_dep, schedule_arr], ignore_index=True).sort_values("time").reset_index(drop=True)

# Áp dụng Luật Tối ưu 2p/0p
MIN_GAP = pd.Timedelta(minutes=2)

for i in range(1, len(schedule)):
    current_row = schedule.iloc[i]
    prev_row = schedule.iloc[i-1]
    
    if current_row["type"] == prev_row["type"]:
        min_time = prev_row["time"] + MIN_GAP
        
        if current_row["time"] < min_time:
            schedule.loc[i, "time"] = min_time
            
schedule["Airport_Info"] = schedule.apply(
    lambda row: f"→{row['Target']}" if row['type'] == 'Takeoff' else f"←{row['Target']}",
    axis=1
)

# THÊM CÁC CỘT THỜI GIAN MỚI ĐỂ ĐỒNG BỘ VỚI WEB VÀ HIỂN THỊ 2 ĐẦU MÚT
# Dùng thời gian tối ưu (time) và duration_hr để tính toán lại 2 đầu mút
schedule['Take_off_time_OPT'] = schedule.apply(
    lambda row: row['time'] if row['type'] == 'Takeoff' else row['time'] - pd.to_timedelta(row['duration_hr'], unit='h'),
    axis=1
)

schedule['Landing_time_PRED'] = schedule.apply(
    lambda row: row['time'] if row['type'] == 'Landing' else row['time'] + pd.to_timedelta(row['duration_hr'], unit='h'),
    axis=1
)

# THÊM CỘT TAKE-OFF DATE
schedule['Take_off_date_OPT'] = schedule['Take_off_time_OPT'].dt.date
schedule['Landing_date_PRED'] = schedule['Landing_time_PRED'].dt.date


# CÁC CỘT CUỐI CÙNG ĐỂ XUẤT RA FILE CSV (Đầy đủ thuộc tính)
FINAL_SCHEDULE_COLS = [
    "Take_off_time_OPT", "Take_off_date_OPT", "Landing_time_PRED", "Landing_date_PRED", 
    "type", "Airport_Info", "duration_hr", 
    'Aircraft', 'Aircraft_type', 'Flight', 'Aircraft_registration',
    'Flight_route', 'Flight_type', 'Flight_purpose', 'Flight_level',
    'Departure_airport', 'Arrival_airport' 
]

schedule[FINAL_SCHEDULE_COLS].to_csv("../source/vvnb_schedule.csv", index=False)
print("✅ Saved optimized VVNB schedule to ../source/vvnb_schedule.csv")

print("7️⃣ Vẽ biểu đồ lịch trình (VVNB)...")
plt.figure(figsize=(14,10)) 
colors={"Takeoff":"tab:blue","Landing":"tab:orange"} 

def get_label(schedule, i, current_type):
    if i == 0:
        return current_type
    prev_type = schedule.iloc[i-1]['type']
    return current_type if current_type != prev_type else ""

for i,row in schedule.iterrows():
    # Sử dụng Take_off_time_OPT cho tọa độ X (đại diện cho sự kiện)
    plt.scatter(row["Take_off_time_OPT"], i, color=colors[row["type"]], 
                label=get_label(schedule, i, row['type']), s=50) 

    plt.text(row["Take_off_time_OPT"], i + 0.2,
             f"{row['Take_off_time_OPT'].strftime('%H:%M')} {row['Airport_Info']} ({row['duration_hr']:.1f}h)", 
             fontsize=7, ha='left')

plt.title(r"Optimized Flight Schedule at Noi Bai Airport (VVNB)" + "\n" + r"Gap: $\geq 2$ min for same operation, 0 min for different operations")
plt.xlabel("Time")
plt.ylabel("Flight Order")
plt.legend(markerscale=2)
plt.gcf().autofmt_xdate() 
plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("../source/vvnb_schedule.png", dpi=300)
print("✅ Saved VVNB schedule chart to ../source/vvnb_schedule.png")
print("8️⃣ Lưu artifacts...")

joblib.dump(model, "../source/flight_duration_model.pkl")
joblib.dump(NUM_FEATURES, "../source/model_features.pkl")
route_stats_dict = {
    "route_mean": route_mean,
    "route_count": route_count,
    "global_mean": global_mean
}
joblib.dump(route_stats_dict, "../source/route_stats.pkl")

print("✅ Training và tối ưu lịch trình VVNB hoàn tất.")