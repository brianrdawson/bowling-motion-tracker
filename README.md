# Bowling Motion Tracker

A wrist-worn IMU motion tracker built using a Seeed Studio XIAO nRF52840 Sense.
The goal is to capture arm swing speed, wrist rotation, and orientation during
bowling throws for analysis and training.

## Project Structure
hardware/ - Firmware for the XIAO board (Arduino)
host/logger/ - Python tools for capturing IMU data
host/analysis/ - Data exploration & visualization notebooks
data/raw/ - Raw CSV logs from IMU sessions
data/processed/ - Cleaned / engineered data
docs/ - Notes, diagrams, ideas
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
