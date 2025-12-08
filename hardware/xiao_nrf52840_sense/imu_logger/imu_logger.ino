#include <Arduino.h>
#include <Wire.h>
#include <LSM6DS3.h>  // Seeed Arduino LSM6DS3 library

// XIAO nRF52840 Sense IMU is at I2C address 0x6A
LSM6DS3 imu(I2C_MODE, 0x6A);

// ---- Sampling params ----
const float SAMPLE_HZ = 200.0;
const unsigned long SAMPLE_PERIOD_US = (unsigned long)(1000000.0 / SAMPLE_HZ);

// ---- Gyro bias (for drift compensation, in dps) ----
float gx_bias = 0.0f;
float gy_bias = 0.0f;
float gz_bias = 0.0f;

// ---- Session control / timing ----
bool isRecording = false;
unsigned long startMicros = 0;
unsigned long nextSampleMicros = 0;

// ---------- Helpers ----------

inline float sqf(float x) { return x * x; }

inline float vecMag(float x, float y, float z) {
  return sqrtf(sqf(x) + sqf(y) + sqf(z));
}

// ---------- Calibration using float gyro reads ----------

void calibrateGyro(uint16_t samples = 500) {
  Serial.println("# Calibrating gyro... hold wrist still.");
  delay(500);

  float gx_sum = 0.0f;
  float gy_sum = 0.0f;
  float gz_sum = 0.0f;

  for (uint16_t i = 0; i < samples; i++) {
    gx_sum += imu.readFloatGyroX();  // dps
    gy_sum += imu.readFloatGyroY();
    gz_sum += imu.readFloatGyroZ();
    delay(2);
  }

  gx_bias = gx_sum / samples;
  gy_bias = gy_sum / samples;
  gz_bias = gz_sum / samples;

  Serial.print("# Gyro bias (dps): ");
  Serial.print(gx_bias); Serial.print(", ");
  Serial.print(gy_bias); Serial.print(", ");
  Serial.println(gz_bias);
}

// ---------- Session helpers ----------

void printCsvHeader() {
  Serial.println(
    "timestamp_ms,"
    "ax,ay,az,"       // accel in g
    "gx,gy,gz,"       // gyro in dps (bias-corrected)
    "accel_mag,gyro_mag"
  );
}

void startSession() {
  isRecording = true;
  Serial.println("# START_SESSION");
  printCsvHeader();

  startMicros = micros();
  nextSampleMicros = startMicros;
}

void stopSession() {
  isRecording = false;
  Serial.println("# END_SESSION");
}

// ---------- Arduino setup / loop ----------

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  Wire.begin();

  if (imu.begin() != 0) {
    Serial.println("! IMU init failed (LSM6DS3)");
    while (1) { delay(100); }
  }

  Serial.println("# Bowling Motion Tracker – Seeed LSM6DS3");
  Serial.println("# Commands: 's' = start, 'e' = end, 'c' = recalibrate gyro");

  calibrateGyro();
}

void loop() {
  // --- Serial commands ---
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 's') {
      if (!isRecording) startSession();
    } else if (c == 'e') {
      if (isRecording) stopSession();
    } else if (c == 'c') {
      if (!isRecording) {
        calibrateGyro();
      } else {
        Serial.println("# Ignoring 'c' while recording");
      }
    }
  }

  if (!isRecording) {
    return;
  }

  // --- Fixed-rate sampling ---
  unsigned long now = micros();
  if ((long)(now - nextSampleMicros) >= 0) {
    nextSampleMicros += SAMPLE_PERIOD_US;

    // Float reads from Seeed library:
    //  - Accel: g
    //  - Gyro: dps
    float ax = imu.readFloatAccelX();
    float ay = imu.readFloatAccelY();
    float az = imu.readFloatAccelZ();

    float gx_raw = imu.readFloatGyroX();
    float gy_raw = imu.readFloatGyroY();
    float gz_raw = imu.readFloatGyroZ();

    // Bias-correct gyro
    float gx = gx_raw - gx_bias;
    float gy = gy_raw - gy_bias;
    float gz = gz_raw - gz_bias;

    float accel_mag = vecMag(ax, ay, az);
    float gyro_mag  = vecMag(gx, gy, gz);

    unsigned long t_ms = (now - startMicros) / 1000UL;

    // CSV line
    Serial.print(t_ms);      Serial.print(",");
    Serial.print(ax, 6);     Serial.print(",");
    Serial.print(ay, 6);     Serial.print(",");
    Serial.print(az, 6);     Serial.print(",");
    Serial.print(gx, 6);     Serial.print(",");
    Serial.print(gy, 6);     Serial.print(",");
    Serial.print(gz, 6);     Serial.print(",");
    Serial.print(accel_mag, 6); Serial.print(",");
    Serial.println(gyro_mag, 6);
  }
}
