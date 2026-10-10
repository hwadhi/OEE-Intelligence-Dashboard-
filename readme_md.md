# OEEiq | Production Intelligence Platform

OEEiq is a next-generation operational analytics platform built for High Pressure Die Casting (HPDC) manufacturing environments. It bridges the gap between shop-floor downtime telemetry and executive financial decision-making by combining real-time OEE (Overall Equipment Effectiveness) tracking, machine learning anomaly detection, and supply chain (SCM) intelligence.

---

## Key Features

- **Real-Time OEE Calculations**: Automatically computes Availability, Performance, Quality, and Overall OEE metrics across shifts and machine lines.
- **Machine Learning Anomaly Engine**: Implements Scikit-Learn Isolation Forest models to flag abnormal shift behaviors and production bottlenecks automatically.
- **Financial Loss Calculator**: Translates micro-downtime and part rejections directly into financial metrics (INR loss estimates based on custom part valuations and cycle speeds).
- **Supply Chain (SCM) Intelligence**: Analyzes supplier On-Time In-Full (OTIF) scorecards and correlates inventory buffer days with production efficiency.
- **Dynamic Theme Engine**: Fully responsive dark and light mode UI architecture with high-contrast data visualizations and clean component styling.
- **Multi-Format Export**: Generates and downloads production-ready CSV reports for shift handovers and procurement audits.

---

## Tech Stack

- **Core Framework**: Python, Streamlit
- **Data Processing**: Pandas, NumPy, SciPy
- **Data Visualization**: Plotly (Interactive Heatmaps, Scatter Timelines, Bar Scorecards)
- **Machine Learning**: Scikit-Learn (Isolation Forest, Standard Scaler)

---

## Project Structure

```text
oee-intelligence-dashboard/
│
├── .streamlit/
│   └── config.toml
├── app.py
├── scm_data.csv (Optional / Custom upload)
├── requirements.txt
└── README.md
```

---

## Local Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/oee-intelligence-dashboard.git
cd oee-intelligence-dashboard
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Streamlit Settings
Ensure your `.streamlit/config.toml` is configured for proper theme rendering:
```toml
[server]
headless = true

[theme]
base = "light"
primaryColor = "#6C63FF"
backgroundColor = "#F8FAFC"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#0F172A"
font = "sans serif"
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## Usage Guide

1. **Landing Portal**: Launch the platform and use the top navigation toggle to switch between Light Mode and Dark Mode instantly. Click **Enter Platform** to access the main dashboard.
2. **Data Input**: The platform runs on a built-in 90-day HPDC sample dataset by default. You can upload custom OEE shift logs or SCM supplier CSV files via the sidebar uploaders.
3. **Filters & Loss Parameters**: Use the sidebar multiselectors to filter by specific machines (`M1-HPDC` through `M4-HPDC`) or shifts (`A`, `B`, `C`). Adjust part values and target percentages to calibrate real-time financial impact calculations.
4. **Modules**: Navigate across tabs to view the OEE Overview heatmap, SCM supplier risk rankings, ML anomaly timelines, and exportable reports.

---

## Author & Contact

**Heramb**  
Production Engineering, VJTI Mumbai ('27)  
Email: hvwadhi_b23@pe.vjti.ac.in  
Platform Version: BETA v2.0