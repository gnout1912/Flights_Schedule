from flask import Flask, request, jsonify
import csv
from datetime import datetime, timedelta
from flask_cors import CORS
from flask import render_template
import io
import pandas as pd
import joblib
import random
import pymysql

app = Flask(__name__)
CORS(app)

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='@wbCDkwT8', 
        database='flight_schedule',
        cursorclass=pymysql.cursors.DictCursor
    )

model = joblib.load('flight_duration_model.pkl')
model_features = joblib.load('model_features.pkl')

MIN_VVNB_GAP = timedelta(minutes=2)
EXTRA_VVNB_GAP_MINUTES = 3

def random_vvnb_gap() -> timedelta:
    """Return a minimum gap plus a random buffer for VVNB operations."""
    extra_minutes = random.uniform(0, EXTRA_VVNB_GAP_MINUTES)
    return MIN_VVNB_GAP + timedelta(minutes=extra_minutes)

def parse_datetime_safe(date_str: str, time_str: str) -> datetime:
    """Parse combined date/time strings coming from CSV in multiple possible formats."""
    if date_str is None or time_str is None:
        raise ValueError("Missing date or time value")

    datetime_str = f"{date_str.strip()} {time_str.strip()}"
    candidate_formats = (
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %H:%M",
    )

    for fmt in candidate_formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue

    for dayfirst in (True, False):
        try:
            return pd.to_datetime(datetime_str, dayfirst=dayfirst).to_pydatetime()
        except (ValueError, TypeError):
            continue

    raise ValueError(
        f"time data '{datetime_str}' does not match supported formats: "
        + ", ".join(candidate_formats)
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File format not supported, upload a CSV file.'}), 400

    try:
        connection = get_connection()
        cursor = connection.cursor()

        file_stream = io.StringIO(file.stream.read().decode('utf-8'))
        csv_data = csv.DictReader(file_stream)
        last_vvnb_event_time = None

        for row in csv_data:
            required_fields = [
                'Take_off_date', 'Take_off_time', 'Flight_route',
                'Departure_airport', 'Arrival_airport'
            ]
            if not all(row.get(key) for key in required_fields):
                cursor.close()
                connection.close()
                return jsonify({'error': 'Missing required fields in the input file.'}), 400

            input_data = pd.DataFrame([{
                'Aircraft': row.get('Aircraft'),
                'Aircraft_type': row.get('Aircraft_type'),
                'Flight_route': row.get('Flight_route'),
                'Flight_type': row.get('Flight_type'),
                'Flight_purpose': row.get('Flight_purpose'),
                'Departure_airport': row.get('Departure_airport'),
                'Arrival_airport': row.get('Arrival_airport'),
                'Flight_level': row.get('Flight_level')
            }])

            input_encoded = pd.get_dummies(input_data)
            input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
            predicted_duration = model.predict(input_encoded)[0]

            take_off_datetime = parse_datetime_safe(row['Take_off_date'], row['Take_off_time'])
            landing_datetime = take_off_datetime + timedelta(hours=predicted_duration)
            if landing_datetime < take_off_datetime:
                landing_datetime += timedelta(days=1)

            if row['Departure_airport'] == 'VVNB':
                if last_vvnb_event_time and take_off_datetime < last_vvnb_event_time + MIN_VVNB_GAP:
                    delta = (last_vvnb_event_time + random_vvnb_gap()) - take_off_datetime
                    take_off_datetime += delta
                    landing_datetime += delta
                last_vvnb_event_time = take_off_datetime

            if row['Arrival_airport'] == 'VVNB':
                if last_vvnb_event_time and landing_datetime < last_vvnb_event_time + MIN_VVNB_GAP:
                    delta = (last_vvnb_event_time + random_vvnb_gap()) - landing_datetime
                    if delta > timedelta(0):
                        landing_datetime += delta
                last_vvnb_event_time = landing_datetime

            cursor.execute("""
                SELECT COUNT(*) AS count FROM flights
                WHERE Flight = %s AND Take_off_date = %s
                AND Departure_airport = %s AND Arrival_airport = %s
            """, (
                row.get('Flight'),
                take_off_datetime.date(),
                row.get('Departure_airport'),
                row.get('Arrival_airport')
            ))
            existing = cursor.fetchone()['count']

            if existing > 0:
                print(f"⚠️ Ignore duplicate flight: {row.get('Flight')} ({row.get('Departure_airport')} → {row.get('Arrival_airport')})")
                continue  

            cursor.execute("""
                INSERT INTO flights (
                    Aircraft, Aircraft_type, Flight, Aircraft_registration,
                    Flight_route, Flight_type, Flight_purpose, Departure_airport,
                    Arrival_airport, Take_off_time, Take_off_date, Landing_time, Landing_date
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row.get('Aircraft'), row.get('Aircraft_type'), row.get('Flight'),
                row.get('Aircraft_registration'), row.get('Flight_route'),
                row.get('Flight_type'), row.get('Flight_purpose'),
                row.get('Departure_airport'), row.get('Arrival_airport'),
                take_off_datetime, take_off_datetime.date(),
                landing_datetime, landing_datetime.date()
            ))

        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'message': 'File uploaded and data saved successfully!'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/flights', methods=['GET'])
def get_flights():
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM flights")
        flights = cursor.fetchall()
        connection.close()
        return jsonify({'flights': flights}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_flight', methods=['POST'])
def delete_flight():
    try:
        flight_id = request.json.get('flight_id')
        if not flight_id:
            return jsonify({'error': 'Missing flight ID'}), 400

        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM flights WHERE id = %s", (flight_id,))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'message': 'Flight deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/flight')
def flight_page():
    return render_template('flight.html')

if __name__ == '__main__':
    app.run(debug=True)
