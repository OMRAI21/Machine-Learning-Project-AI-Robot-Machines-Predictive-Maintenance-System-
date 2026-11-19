# Robot Health Monitor: Predictive Maintenance for Industrial Robots

A machine-learning project that predicts when industrial robots are likely to fail, estimates how many days of life remain, and identifies the most likely failure cause. Results are shown in an interactive web dashboard.

## Background

This project builds on existing research showing that AI and machine-learning models can predict equipment failures from sensor data such as vibration, temperature and motor current, reducing unplanned downtime and maintenance costs [1-4]. Following the Random Forest approach used in this literature [2], this project applies it to industrial robots and extends it with remaining-useful-life estimation and failure-mode diagnosis in an interactive dashboard.

## Features

- **Failure risk prediction**: probability that a robot will fail within the next **14 days**.
- **Remaining Useful Life (RUL) estimate**: predicted days to failure, capped at 120 ("120+").
- **Failure mode diagnosis**: classifies the likely cause as bearing wear, gearbox wear, motor overheat or lubrication loss.
- **Maintenance queue**: robots are ranked by urgency.
- **Manual input**: enter sensor readings and get an instant prediction.
- **Batch analysis**: upload a CSV with many robots' readings.
- **Explainability**: shows which sensors drive the failure risk.

## Project Structure

| File | Description |
|---|---|
| `train_model.py` | Simulates fleet data, trains the models, evaluates them and exports the results |
| `Robot_Health_Monitor.html` | Interactive dashboard (open in any browser) |
| `model.json` | Trained Random Forest models exported as JSON |
| `fleet.json` | Sensor history for 12 demo robots shown in the dashboard |
| `metrics.json` | Evaluation metrics on held-out test robots |
| `sample_readings.csv` | Example input file for the batch analysis feature |
| `maintenance_system.db` | SQLite database with sample sensor readings, maintenance tasks and predictions |
| `citation-391673848.ris` | Reference file for the related research paper |

## How It Works

1. **Data simulation**: `train_model.py` simulates 420 robots running from new until failure. Each robot has one of four failure modes, and sensor readings degrade along a realistic wear curve (slow early, fast late) with noise.
2. **Features (11)**: motor temperature, gearbox oil temperature, vibration, motor current, torque ripple, position error, oil particle count, load, ambient temperature, operating hours and days since service.
3. **Models (Random Forest)**:
   - Classifier for "will fail within 14 days"
   - Regressor for remaining useful life
   - Classifier for failure mode
4. **Evaluation**: the data is split by robot (`GroupShuffleSplit`), so the test set contains only robots the model has never seen. This avoids data leakage.
5. **Export**: the trained forests are exported as JSON so the dashboard can run predictions directly in the browser.

## Results (held-out test robots)

| Metric | Value |
|---|---|
| Failure risk AUC | 0.9924 |
| Recall (14-day window) | 0.7925 |
| Precision (14-day window) | 0.7736 |
| RUL MAE (all) | 7.25 days |
| RUL MAE (last 60 days of life) | 9.13 days |
| Failure mode accuracy | 99.77% |

> **Note:** The data is **simulated**, not collected from real robots. The high scores reflect clean, synthetic failure patterns, and real factory data would be noisier and give lower scores. Validating on real sensor data is future work.

## Getting Started

### Requirements

- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`

```bash
pip install numpy pandas scikit-learn
```

### Retrain the models

```bash
python train_model.py
```

This prints the metrics and regenerates `model.json`, `fleet.json`, `metrics.json` and `sample_readings.csv`.

### Open the dashboard

Open `Robot_Health_Monitor.html` in a web browser. To try batch analysis, upload `sample_readings.csv`.

## Database

`maintenance_system.db` is a SQLite database with sample data:

- `sensor_readings`: temperature, vibration, pressure, current, voltage, speed, load and operating hours per robot
- `maintenance_tasks`: task type, priority, scheduled time, required parts and status
- `predictions`: failure probability, predicted failure time, recommended action, confidence and anomaly score

## Tech Stack

Python · NumPy · pandas · scikit-learn · SQLite · HTML/CSS/JavaScript

## Limitations and Future Work

- Train and validate on real robot sensor data.
- Add more failure modes and robot types.
- Connect live sensor streams to the dashboard.
- Tune the alert threshold to balance missed failures against false alarms.

## References

1. D. Kummari, "AI-Driven Predictive Maintenance for Industrial Robots in Automotive Manufacturing: A Case Study," *International Journal of Scientific Research and Modern Technology*, vol. 1, no. 12, pp. 107-119, Dec. 2022. doi: 10.38124/ijsrmt.v1i12.489

2. A. Mule, P. Bhoirkar, A. Jadhav, P. Deshmukh, and L. More, "An Integrated IoT-AI Framework for Real-Time Predictive Maintenance in Industrial Machinery," *International Journal of Scientific Research in Engineering and Management (IJSREM)*, vol. 10, no. 5, May 2026. https://ijsrem.com

3. S. Zeb and S. K. Lodhi, "AI for Predictive Maintenance: Reducing Downtime and Enhancing Efficiency," *Enrichment: Journal of Multidisciplinary Research and Development*, vol. 3, no. 1, pp. 135-150, 2025. doi: 10.55324/enrichment.v3i1.338

4. I. B. Demir, "Artificial Intelligence for Predictive Maintenance," M.Sc. thesis (Management Engineering), Politecnico di Milano, Advisor: G. Miragliotta, Academic Year 2023-24.

## Author

Divyansh Rai · RA2411003030016 · B.Tech Computer Science Engineering
