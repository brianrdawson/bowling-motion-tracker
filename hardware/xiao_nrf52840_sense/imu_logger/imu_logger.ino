#include <Wire.h>
#include "LSM6DS3.h"

// I2C mode, address 0x6A is default for XIAO nRF52840 Sense IMU
LSM6DS3 myIMU(I2C_MODE, 0x6A);

unsigned long lastTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 10;  // ~100 Hz

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for USB serial
  }

  if (myIMU.begin() != 0) {
    Serial.println("Failed to initialize IMU!");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("t_ms,ax_mps2,ay_mps2,az_mps2,gx_dps,gy_dps,gz_dps");
}

void loop() {
  unsigned long now = millis();
  if (now - lastTime >= SAMPLE_INTERVAL_MS) {
    lastTime = now;

    float ax = myIMU.readFloatAccelX();
    float ay = myIMU.readFloatAccelY();
    float az = myIMU.readFloatAccelZ();

    float gx = myIMU.readFloatGyroX();
    float gy = myIMU.readFloatGyroY();
    float gz = myIMU.readFloatGyroZ();

    Serial.print(now);
    Serial.print(",");
    Serial.print(ax); Serial.print(",");
    Serial.print(ay); Serial.print(",");
    Serial.print(az); Serial.print(",");
    Serial.print(gx); Serial.print(",");
    Serial.print(gy); Serial.print(",");
    Serial.println(gz);
  }
}
