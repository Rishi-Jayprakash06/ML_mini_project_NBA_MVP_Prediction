# NBA MVP Prediction Using Machine Learning

## Project Overview

This project explores the use of machine learning to predict the NBA Most Valuable Player (MVP) using regular-season player statistics. It compares three classification algorithms and evaluates their ability to rank MVP candidates.

## Objectives

- Analyze historical NBA player statistics and MVP voting records.
- Prepare and preprocess the datasets for machine learning.
- Train and compare Logistic Regression, Naive Bayes, and Random Forest models.
- Evaluate model performance using ranking-based metrics.
- Identify the player statistics that contribute most to the Random Forest model's predictions.

## Dataset

The project uses two CSV files:

- `Player Award Shares.csv` — contains historical NBA award voting information.
- `Player Per Game.csv` — contains player statistics for individual seasons.

The datasets cover NBA seasons from 1956 to 2025. Data is joined using player identifiers and seasons.

## Machine Learning Models

1. **Logistic Regression** — a linear classification baseline.
2. **Gaussian Naive Bayes** — a probabilistic classification model.
3. **Random Forest** — an ensemble of decision trees.

## Evaluation

The models are compared using average rank, median rank, Top-1 accuracy, and Top-3 accuracy.

In the current evaluation, Random Forest achieves an average rank of 2.333 and Top-1 accuracy of 50%. All three models achieve Top-3 accuracy of approximately 83.3%.

These results are based on six test seasons from 2020 to 2025 and should be interpreted in the context of the evaluation method used.

## Feature Importance

The Random Forest feature-importance analysis identifies points per game and minutes played per game as the two most influential features in the current model.

## Project Structure

```text
NBA-MVP-ML/
├── data/
│   ├── Player Award Shares.csv
│   └── Player Per Game.csv
├── models/
│   ├── rf_full.joblib
│   ├── logreg_full.joblib
│   ├── nb_full.joblib
│   ├── imputer_full.joblib
│   └── scaler_full.joblib
├── results/
│   ├── model_comparison.png
│   └── feature_importance.png
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── app.py                     # Flask Web GUI Server
├── gui_desktop.py             # Native Desktop GUI (Tkinter)
├── model_service.py           # ML Model Engine & Inference API
├── NBA_MVP_Prediction.ipynb   # Analysis & Training Notebook
├── requirements.txt
└── README.md
```

## Interactive GUI (Web & Desktop)

The project includes an interactive graphical user interface (GUI) supporting manual stat input, explicit feature limits, historical NBA averages, real-world examples, and immediate classification of whether a player is an **MVP or NOT**.

### Option 1: Web GUI (Modern Dark Arena Aesthetic)

Launch the web application locally:

```bash
python app.py
```

Then open your browser at **`http://127.0.0.1:5000`**.

Features:
- **Manual Numerical Entry & Dual Range Sliders**: Type exact stats or adjust sliders with live sync.
- **Strict Limits & Benchmark References**: Each feature displays allowable limits, NBA league averages, and historical MVP winner ranges.
- **Historical Player Presets**: One-click autofill for iconic MVP seasons (Nikola Jokić 2024, Shai Gilgeous-Alexander 2025, Joel Embiid 2023, Stephen Curry 2016, Michael Jordan 1996, Giannis 2020) and non-MVP contenders / role players.
- **Clear MVP Output**: Instant verdict banner (**🏆 PREDICTED MVP: YES** vs. **❌ PREDICTED MVP: NO**) with confidence percentage.
- **Consensus Breakdown**: Compares predictions across Random Forest, Logistic Regression, and Gaussian Naive Bayes.
- **Scouting Analytics**: Visual stat delta cards against 70 historical MVPs and Random Forest feature importance weights.

### Option 2: Native Desktop GUI (Tkinter)

Launch the standalone desktop application window:

```bash
python gui_desktop.py
```

## How to Run

1. Install Python 3.11 or a compatible version.
2. Open a terminal in the project directory.
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Web GUI:
   ```bash
   python app.py
   ```
   Or launch the Desktop GUI:
   ```bash
   python gui_desktop.py
   ```
5. Or run the Jupyter Notebook:
   Open `NBA_MVP_Prediction.ipynb` in Jupyter Notebook or VS Code and run all cells.

## Results

The project generates two visualizations:

- `results/model_comparison.png` — compares model ranking accuracy.
- `results/feature_importance.png` — displays Random Forest feature importance.

## Limitations

- The evaluation covers only six test seasons.
- MVP voting is influenced by factors beyond basic per-game statistics.
- Historical patterns may not generalize to future seasons.
- Feature importance indicates predictive contribution, not causation.

## Data Sources

- Basketball-Reference: https://www.basketball-reference.com/
- Kaggle NBA/ABA/BAA statistics dataset: https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats

This project is intended for educational and experimental purposes.

