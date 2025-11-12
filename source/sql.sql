CREATE DATABASE IF NOT EXISTS flight_schedule;
USE flight_schedule;

CREATE TABLE flights (
    id INT AUTO_INCREMENT PRIMARY KEY,

    Aircraft VARCHAR(20),                
    Aircraft_type VARCHAR(5),            
    Flight VARCHAR(20),                  
    Aircraft_registration VARCHAR(20),   
    Flight_route VARCHAR(20),            
    Flight_type VARCHAR(10),             
    Flight_purpose VARCHAR(10),          

    Departure_airport VARCHAR(10),      
    Arrival_airport VARCHAR(10),        

    Flight_level INT,                    

    Take_off_time DATETIME,              
    Take_off_date DATE,                  
    Landing_time DATETIME,               
    Landing_date DATE,                  

    delay_time VARCHAR(20) DEFAULT NULL, 
    reason VARCHAR(255) DEFAULT NULL,    

    UNIQUE KEY unique_flight (Flight, Take_off_date, Departure_airport, Arrival_airport)
);
