import pandas as pd
from datetime import datetime
from bisect import bisect_right
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

file_path = 'train_file.csv'
data = pd.read_csv(file_path)

def parse_datetime(datetime_str):
    for fmt in ("%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Datetime '{datetime_str}' does not match expected formats %d/%m/%Y %%H:%M:%S or %m/%d/%Y %%H:%M:%S")


def calculate_flight_duration(row):
    take_off = parse_datetime(f"{row['Take_off_date']} {row['Take_off_time']}")
    landing = parse_datetime(f"{row['Landing_date']} {row['Landing_time']}")
    duration = (landing - take_off).total_seconds() / 3600
    return duration


def enrich_flight_record(flight):
    take_off_dt = parse_datetime(f"{flight['Take_off_date']} {flight['Take_off_time']}")
    landing_dt = parse_datetime(f"{flight['Landing_date']} {flight['Landing_time']}")
    return {
        **flight,
        'take_off_dt': take_off_dt,
        'landing_dt': landing_dt
    }

data['Flight_duration'] = data.apply(calculate_flight_duration, axis=1)

X = data[['Aircraft', 'Aircraft_type', 'Flight_route', 'Flight_type',
          'Flight_purpose', 'Departure_airport', 'Arrival_airport', 'Flight_level']]
y = data['Flight_duration']

X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
median_ae = np.median(np.abs(y_test - y_pred))
print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
print(f"Median Absolute Error (MedAE): {median_ae:.2f} hours")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} hours")
print(f"R-squared (R²): {r2:.3f}")

def greedy_schedule_flights(flights):
    flights_sorted = sorted(flights, key=lambda x: x['take_off_dt'])
    optimized_schedule = []
    last_landing_time = None

    for flight in flights_sorted:
        if last_landing_time is None or flight['take_off_dt'] >= last_landing_time:
            optimized_schedule.append(flight)
            last_landing_time = flight['landing_dt']

    return optimized_schedule

def backtracking_schedule(flights):
    flights_sorted = sorted(flights, key=lambda x: x['take_off_dt'])
    weights = [flight['Flight_duration'] for flight in flights_sorted]
    n = len(flights_sorted)

    @lru_cache(maxsize=None)
    def solve(current_index, last_selected_index):
        if current_index == n:
            return 0.0, ()

        best_weight, best_indices = solve(current_index + 1, last_selected_index)

        last_landing_time = (
            flights_sorted[last_selected_index]['landing_dt']
            if last_selected_index != -1
            else None
        )
        flight = flights_sorted[current_index]

        if last_landing_time is None or flight['take_off_dt'] >= last_landing_time:
            include_weight, include_indices = solve(current_index + 1, current_index)
            include_weight += weights[current_index]
            include_indices = (current_index,) + include_indices

            if include_weight > best_weight:
                best_weight, best_indices = include_weight, include_indices

        return best_weight, best_indices

    _, optimal_indices = solve(0, -1)
    return [flights_sorted[i] for i in optimal_indices]

def csp_schedule(flights):
    flights_sorted = sorted(flights, key=lambda x: x['take_off_dt'])
    assigned_flights = []

    for flight in flights_sorted:
        if all(flight['take_off_dt'] >= assigned['landing_dt'] for assigned in assigned_flights):
            assigned_flights.append(flight)

    return assigned_flights

def dynamic_programming_schedule(flights):
    flights_sorted = sorted(flights, key=lambda x: x['landing_dt'])
    n = len(flights_sorted)

    if n == 0:
        return []

    end_times = [flight['landing_dt'] for flight in flights_sorted]
    predecessors = []
    for i in range(n):
        take_off_time = flights_sorted[i]['take_off_dt']
        j = bisect_right(end_times, take_off_time) - 1
        predecessors.append(j)

    dp = [0.0] * (n + 1)
    decision = [False] * n

    for i in range(1, n + 1):
        flight = flights_sorted[i - 1]
        pred_index = predecessors[i - 1] + 1 if predecessors[i - 1] != -1 else 0
        include_value = flight['Flight_duration'] + dp[pred_index]
        exclude_value = dp[i - 1]
        if include_value >= exclude_value:
            dp[i] = include_value
            decision[i - 1] = True
        else:
            dp[i] = exclude_value

    schedule = []
    i = n
    while i > 0:
        if decision[i - 1]:
            schedule.append(flights_sorted[i - 1])
            i = predecessors[i - 1] + 1
        else:
            i -= 1

    schedule.reverse()
    return schedule

flights_data = data.to_dict('records')
flights_enriched = [enrich_flight_record(flight) for flight in flights_data]


def summarize_schedule(name, schedule, sample_size=5):
    total_duration = sum(flight['Flight_duration'] for flight in schedule)
    print(f"{name}: {len(schedule)} flights, total duration {total_duration:.2f} hours")
    sample = [
        {
            'Flight': flight['Flight'],
            'Departure': f"{flight['Departure_airport']} ({flight['take_off_dt'].strftime('%d/%m/%Y %H:%M')})",
            'Arrival': f"{flight['Arrival_airport']} ({flight['landing_dt'].strftime('%d/%m/%Y %H:%M')})",
            'Duration_h': round(flight['Flight_duration'], 2)
        }
        for flight in schedule[:sample_size]
    ]
    print(f"  Sample: {sample}")

optimized_schedule_greedy = greedy_schedule_flights(flights_enriched)
optimized_schedule_backtracking = backtracking_schedule(flights_enriched)
optimized_schedule_csp = csp_schedule(flights_enriched)
optimized_schedule_dp = dynamic_programming_schedule(flights_enriched)

summarize_schedule("Optimized Schedule (Greedy)", optimized_schedule_greedy)
summarize_schedule("Optimized Schedule (Backtracking)", optimized_schedule_backtracking)
summarize_schedule("Optimized Schedule (CSP)", optimized_schedule_csp)
summarize_schedule("Optimized Schedule (Dynamic Programming)", optimized_schedule_dp)

joblib.dump(model, '../source/flight_duration_model.pkl')
joblib.dump(X_encoded.columns, '../source/model_features.pkl')

