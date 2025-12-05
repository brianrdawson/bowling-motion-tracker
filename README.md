# Bowling Motion Tracker

A wrist-worn IMU motion tracker built using a Seeed Studio XIAO nRF52840 Sense.
The goal is to capture arm swing speed, wrist rotation, and orientation during
bowling throws for analysis and training.

## Project Structure
hardware/ - Firmware for the XIAO board (Arduino)<br>
host/logger/ - Python tools for capturing IMU data<br>
host/analysis/ - Data exploration & visualization notebooks<br>
data/raw/ - Raw CSV logs from IMU sessions<br>
data/processed/ - Cleaned / engineered data<br>
docs/ - Notes, diagrams, ideas<br>

## Requirements
- Arduino IDE (for firmware)
- Python 3.9+
- `pip install -r requirements.txt`

## Roadmap

- [x] Initial IMU CSV logger (Arduino)
- [ ] Serial data capture (Python)
- [ ] Swing speed analysis
- [ ] Wrist rotation detection
- [ ] Orientation estimation (Madgwick filter)
- [ ] BLE streaming mode
- [ ] Wristband enclosure + mount
