from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import csv, io, math
from datetime import datetime, timedelta, timezone
import pandas as pd
import pymysql
import joblib
import numpy as np 

app = Flask(__name__)
CORS(app)

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="flight_schedule",
        cursorclass=pymysql.cursors.DictCursor,
        init_command="SET time_zone = '+07:00'"   
    )

MODEL = joblib.load('flight_duration_model.pkl')     
NUM_FEATURES = joblib.load('model_features.pkl')     
ROUTE_STATS = joblib.load('route_stats.pkl')         
ROUTE_MEAN   = ROUTE_STATS["route_mean"] 
ROUTE_COUNT  = ROUTE_STATS["route_count"]
GLOBAL_MEAN  = ROUTE_STATS["global_mean"]

MIN_VVNB_GAP = timedelta(minutes=2)

AIRPORT_XY = {
    "VVNB": (21.221,105.807), "VVTS": (10.818,106.652), "VVDN": (16.043,108.199),
    "VVCT": (10.085,105.711), "VVCA": (15.403,108.706), "VVCR": (12.007,109.219),
    "VVDB": (21.385,103.018), "VVDL": (11.75,108.37),  "VVPQ": (10.227,103.96),
    "VVPC": (13.954,109.041), "VVPB": (17.514,106.590),"VVPK": (9.962,105.133),
    "VVTH": (20.824,106.724), "RJAA": (35.773,140.392),"RJTT": (35.552,139.779),
    "RKSI": (37.469,126.451), "ZBAA": (40.080,116.585),"ZSPD": (31.144,121.807),
    "ZGGG": (23.392,113.299), "ZGSZ": (22.639,113.811),"WSSS": (1.364,103.991),
    "WMKK": (2.7456,101.709), "VTBS": (13.690,100.750), "VHHH": (22.308,113.918),
}

def haversine_km(a, b):
    (lat1, lon1), (lat2, lon2) = a, b
    R = 6371.0
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    aa = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(aa))

from datetime import timezone, timedelta

def _normalize_time(t: str) -> str:
    t = str(t).strip()
    if not t:
        return t
    parts = t.split(":")
    if len(parts) == 3:
        parts[0] = parts[0].zfill(2)
        return ":".join(parts)
    return t

def parse_dt(date_str, time_str):
    date_s = str(date_str).strip()
    time_s = _normalize_time(time_str)

    dt = pd.to_datetime(
        f"{date_s} {time_s}",
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce"
    )
    if pd.isna(dt):
        raise ValueError(f"Unparseable datetime: {date_s} {time_s}. Format MM/DD/YYYY expected.")

    if isinstance(dt, pd.Timestamp) and dt.tzinfo is None:
        dt = dt.tz_localize("Asia/Ho_Chi_Minh")
    elif isinstance(dt, pd.Timestamp):
        dt = dt.tz_convert("Asia/Ho_Chi_Minh")

    return dt.tz_localize(None)


def is_domestic(dep, arr):
    return str(dep).startswith("VV") and str(arr).startswith("VV")

def parse_level(x):
    try:
        if isinstance(x, str) and "/" in x:
            vals = [float(v.strip()) for v in x.split("/") if v.strip()]
            return sum(vals) / len(vals) if vals else 0.0
        return float(x)
    except:
        return 0.0

def route_baseline(dep, arr, alpha=30.0):
    key = f"{dep}-{arr}" 
    if key in ROUTE_MEAN:
        n  = ROUTE_COUNT.get(key, 0.0)
        mu = ROUTE_MEAN[key]
        return (n*mu + alpha*GLOBAL_MEAN) / (n + alpha)
    return GLOBAL_MEAN

def build_features(row):
    dep, arr = row.get("Departure_airport"), row.get("Arrival_airport")
    dist_km = 0.0
    if dep in AIRPORT_XY and arr in AIRPORT_XY:
        dist_km = haversine_km(AIRPORT_XY[dep], AIRPORT_XY[arr])

    takeoff = parse_dt(row["Take_off_date"], row["Take_off_time"])

    dep_hour = takeoff.hour
    dep_minute = takeoff.minute
    dep_wd   = takeoff.weekday()
    flight_level = parse_level(row.get("Flight_level")) 

    feats_df = pd.DataFrame([{
        "takeoff_hour": dep_hour,
        "takeoff_minute": dep_minute,
        "takeoff_dayofweek": dep_wd,
        "Flight_level": flight_level, 
        "Departure_airport": dep,
        "Arrival_airport": arr,
    }])

    feats_df = pd.get_dummies(feats_df, columns=["Departure_airport", "Arrival_airport"], prefix=["dep", "arr"])
    
    route_bl = route_baseline(dep, arr)

    feats_df["route_baseline"] = route_bl
    feats = feats_df.reindex(columns=NUM_FEATURES, fill_value=0.0).astype(float)
    
    return feats, takeoff, dist_km

def clamp_duration(dep, arr, pred_hr, dist_km):
    if dist_km and dist_km > 0:
        min_hr, max_hr = dist_km/1000.0, dist_km/300.0
        pred_hr = max(min(pred_hr, max_hr), min_hr)
    
    mean = ROUTE_MEAN.get(f"{dep}-{arr}", GLOBAL_MEAN)
    pred_hr = max(min(pred_hr, mean+2.0), mean-2.0)
    
    return float(max(min(pred_hr, 18.0), 0.3))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if not f.filename.endswith(".csv"):
        return jsonify({'error': 'Upload CSV file.'}), 400

    conn = get_connection()
    cur  = conn.cursor()

    reader = csv.DictReader(io.StringIO(f.stream.read().decode('utf-8')))
    inserted = skipped = 0

    for row in reader:
        try:
            X, takeoff_dt, dist_km = build_features(row)
        except Exception as e:
            print(f"Skip row (parse error: {e}) for flight: {row.get('Flight', 'N/A')}")
            continue

        dep, arr = row.get("Departure_airport"), row.get("Arrival_airport")
        
        base_hr = route_baseline(dep, arr) 
        
        residual_min = float(MODEL.predict(X)[0])
        residual_hr = residual_min / 60.0
        
        pred_hours = clamp_duration(dep, arr, base_hr + residual_hr, dist_km)

        landing_dt = takeoff_dt + timedelta(hours=pred_hours)
        
        cur.execute("""
            SELECT COUNT(*) AS c FROM flights
            WHERE COALESCE(Take_off_time, '1000-01-01') = %s
              AND COALESCE(Landing_time, '1000-01-01') = %s
              AND Departure_airport=%s AND Arrival_airport=%s
        """, (takeoff_dt, landing_dt, dep, arr))
        if cur.fetchone()["c"] > 0:
            skipped += 1
            continue
        
        cur.execute("""
            INSERT INTO flights (
                Aircraft, Aircraft_type, Flight, Aircraft_registration,
                Flight_route, Flight_type, Flight_purpose,
                Departure_airport, Arrival_airport,
                Take_off_time, Take_off_date, Landing_time, Landing_date
                )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
            Take_off_time = VALUES(Take_off_time),
            Landing_time = VALUES(Landing_time),
            Take_off_date = VALUES(Take_off_date),
            Landing_date = VALUES(Landing_date)
        """, (
            row.get("Aircraft"), row.get("Aircraft_type"), row.get("Flight"),
            row.get("Aircraft_registration"), row.get("Flight_route"),
            row.get("Flight_type"), row.get("Flight_purpose"),
            dep, arr,
            takeoff_dt.strftime("%Y-%m-%d %H:%M:%S"), 
            takeoff_dt.date(),                         
            landing_dt.strftime("%Y-%m-%d %H:%M:%S"),  
            landing_dt.date()                          
        ))

        inserted += 1

    conn.commit()
    cur.close(); conn.close()
    return jsonify({"message":"OK","inserted":inserted,"skipped":skipped}), 200

@app.route('/vvnb_schedule', methods=['GET'])
def get_vvnb_schedule():
    try:
        schedule_df = pd.read_csv("../source/vvnb_schedule.csv")
        
        schedule_df['Take_off_time_OPT'] = pd.to_datetime(schedule_df['Take_off_time_OPT'])
        schedule_df['Landing_time_PRED'] = pd.to_datetime(schedule_df['Landing_time_PRED'])
        
        schedule_df = schedule_df.replace({np.nan: None, pd.NaT: None})
        
        schedule_list = schedule_df.to_dict('records')

        return jsonify({"schedule": schedule_list}), 200

    except FileNotFoundError:
        return jsonify({"error": "File vvnb_schedule.csv not found. Please run train_model.py first."}), 500
    except Exception as e:
        return jsonify({"error": f"INTERNAL SERVER ERROR: {str(e)}"}), 500


@app.route('/flights', methods=['GET'])
def get_flights():
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT * FROM flights ORDER BY COALESCE(Take_off_time, Landing_time)")
    flights = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"flights": flights}), 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/flight')
def flight_page():
    return render_template('flight.html')

if __name__ == '__main__':
    app.run(debug=True)
