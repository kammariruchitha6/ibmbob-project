# 🏏 IPL Match Winner Predictor

A full-stack machine learning application that predicts the winner of IPL cricket matches using historical data from **2008 to 2026**, powered by a **Random Forest Classifier** and an interactive **Streamlit** web interface.

---

## 📁 Project Structure

```
ipl_prediction/
├── app.py                            # Main app — ML backend + Streamlit frontend
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── IPL_Match_Prediction_Report.docx  # Detailed project report
├── ipl_rf_model.pkl                  # Saved model (auto-generated on first run)
└── ipl_encoders.pkl                  # Saved encoders (auto-generated on first run)

../
└── IPL_Matches_Data_2008_2026.csv    # Dataset (must be one level above ipl_prediction/)
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python **3.10** or higher
- `pip` package manager

### 2. Install dependencies

```bash
cd ipl_prediction
pip install -r requirements.txt
```

### 3. Place the dataset

Ensure `IPL_Matches_Data_2008_2026.csv` is in the **parent directory** of `ipl_prediction/`:

```
Desktop/IBM/
├── IPL_Matches_Data_2008_2026.csv   ← here
└── ipl_prediction/
    └── app.py
```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The model trains automatically on the first launch (~5–10 seconds) and is cached for all subsequent runs.

---

## 🧠 How It Works

### Data Pipeline

```
CSV  ──►  Filter (complete matches)  ──►  Drop nulls  ──►  Label Encode  ──►  Train/Test Split
```

### Model

| Parameter         | Value                  |
|-------------------|------------------------|
| Algorithm         | Random Forest Classifier |
| Estimators        | 300 trees              |
| Max Depth         | 12                     |
| Class Weight      | balanced               |
| Train/Test Split  | 80% / 20%              |
| Random State      | 42                     |

### Features Used (pre-match only)

| Feature        | Description                        |
|----------------|------------------------------------|
| `team1`        | First-listed team                  |
| `team2`        | Second-listed team                 |
| `toss_winner`  | Team that won the coin toss        |
| `toss_decision`| `bat` or `field`                   |
| `city`         | Host city (venue proxy)            |

**Target:** `winner` — the team that won the match.

> ⚠️ Post-match features (runs, wickets, margins) are deliberately excluded to prevent data leakage.

---

## 🖥️ App Pages

### 🔮 Predict Match
- Select **Team 1**, **Team 2**, **Toss Winner**, **Toss Decision**, and **City**
- Click **Predict Winner** to get the predicted team + win probability bar chart
- View **Head-to-Head History** pie chart between the selected teams

### 📊 Data Insights
- **Most Wins** — All-time win counts by team (bar chart)
- **Toss Analysis** — Win rate by toss decision (bat vs field)
- **Venue Stats** — Top 15 host cities by number of matches
- **Season Trend** — Matches played per season (2008–2026)

### 🤖 Model Performance
- Test Accuracy score
- Feature Importances chart
- Confusion Matrix heatmap (top 10 teams)
- Classification Report table

### 📋 Dataset Explorer
- Filterable table of all 1,000+ completed matches
- Filter by season and/or team

---

## 📦 Dependencies

| Package       | Version  | Role                          |
|---------------|----------|-------------------------------|
| streamlit     | 1.35.0   | Web UI framework              |
| pandas        | 2.2.2    | Data manipulation             |
| numpy         | 1.26.4   | Numerical computing           |
| scikit-learn  | 1.5.0    | ML model + metrics            |
| matplotlib    | 3.9.0    | Confusion matrix plot         |
| seaborn       | 0.13.2   | Heatmap styling               |
| plotly        | 5.22.0   | Interactive charts            |
| joblib        | 1.4.2    | Model/encoder serialisation   |

---

## 📊 Dataset

**File:** `IPL_Matches_Data_2008_2026.csv`

- **Source:** https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
- **Seasons:** 2007/08 through 2026
- **Rows:** ~1,100+ completed matches
- **Columns:** 30 (team info, toss details, scores, officials, players)

Key columns used:

```
event_name, season, date, city, venue,
team1, team2, toss_winner, toss_decision,
winner, result_type, win_by_runs, win_by_wickets,
player_of_match
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Frontend                │
│  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌────────┐  │
│  │ Predict  │  │ Insights │  │ Model│  │ Data   │  │
│  │  Match   │  │  Charts  │  │ Eval │  │Explorer│  │
│  └────┬─────┘  └──────────┘  └──────┘  └────────┘  │
│       │                                             │
│  ┌────▼──────────────────────────────────────────┐  │
│  │            Python ML Backend                  │  │
│  │  pandas → LabelEncoder → RandomForest         │  │
│  │  predict_proba → Win Probabilities            │  │
│  └───────────────────────────────────────────────┘  │
│       │                                             │
│  ┌────▼────────────────────────────────────────┐    │
│  │   IPL_Matches_Data_2008_2026.csv            │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 📈 Sample Results

| Metric        | Value (approx.) |
|---------------|-----------------|
| Test Accuracy | ~55–65%         |
| Top Feature   | team1 / team2   |
| Toss Win Rate | ~50–52%         |

> T20 cricket is inherently unpredictable. An accuracy of 55–65% significantly beats random chance (50%) given only 5 pre-match features.

---

## 🔮 Future Enhancements

- [ ] Add player-level features (batting average, bowling economy)
- [ ] Mid-innings score predictor (live match state)
- [ ] Compare XGBoost / Neural Network models
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Add One-Hot Encoding instead of Label Encoding for city/team

---

## 📄 License

This project is for educational and research purposes. IPL data is publicly available through official records.

---

## 🙏 Acknowledgements

- **BCCI / IPL** https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
- **Scikit-learn** team for the ML framework
- **Streamlit** for making Python web apps effortless
- **Plotly** for beautiful interactive visualisations
