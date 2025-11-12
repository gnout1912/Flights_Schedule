CREATE DATABASE IF NOT EXISTS flight_schedule;
USE flight_schedule;

CREATE TABLE flights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Aircraft VARCHAR(50),
    Aircraft_type VARCHAR(10),
    Flight VARCHAR(50),
    Aircraft_registration VARCHAR(50),
    Flight_route VARCHAR(50),
    Flight_type VARCHAR(10),
    Flight_purpose VARCHAR(10),
    Departure_airport VARCHAR(10),
    Arrival_airport VARCHAR(10),
    Take_off_time DATETIME,
    Take_off_date DATE,
    Landing_time DATETIME,
    Landing_date DATE
);