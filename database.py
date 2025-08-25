-- Step 1: Create Database
CREATE DATABASE forensic_fingerprint;
USE forensic_fingerprint;

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,   -- Unique ID for each user
    name VARCHAR(100) NOT NULL,               -- User's name
    fingerprint LONGBLOB NOT NULL,            -- Fingerprint template (binary data)
    photo LONGBLOB,                           -- Photo storage (binary data)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Record creation time
);
describe users ;
