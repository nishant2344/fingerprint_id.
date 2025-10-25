CREATE DATABASE fingerprints_db;
USE fingerprints_db;

CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    fingerprint LONGBLOB NOT NULL,
    minutiae LONGBLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE students
  MODIFY fingerprint LONGBLOB NULL DEFAULT NULL;
