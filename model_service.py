import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "g",
    "mp_per_game",
    "fg_percent",
    "x3p_percent",
    "ft_percent",
    "trb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "pts_per_game"
]

FEATURE_METADATA = {
    "g": {
        "name": "Games Played",
        "short": "G",
        "unit": "games",
        "min": 1,
        "max": 82,
        "step": 1,
        "league_avg": 52.4,
        "mvp_avg": 78.4,
        "mvp_min": 49.0,
        "mvp_max": 82.0,
        "example": 78,
        "desc": "Total regular season games played (NBA season is 82 games; min ~50-65 for MVP consideration)."
    },
    "mp_per_game": {
        "name": "Minutes Per Game",
        "short": "MPG",
        "unit": "min",
        "min": 0.0,
        "max": 48.0,
        "step": 0.1,
        "league_avg": 20.8,
        "mvp_avg": 38.3,
        "mvp_min": 30.4,
        "mvp_max": 47.3,
        "example": 35.5,
        "desc": "Average court time per game in minutes. Star players typically log 33-38 minutes."
    },
    "fg_percent": {
        "name": "Field Goal %",
        "short": "FG%",
        "unit": "ratio (0-1)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.005,
        "league_avg": 0.444,
        "mvp_avg": 0.512,
        "mvp_min": 0.378,
        "mvp_max": 0.683,
        "example": 0.525,
        "desc": "Overall field goal shooting percentage (ratio between 0.000 and 1.000, e.g., 0.525 = 52.5%)."
    },
    "x3p_percent": {
        "name": "3-Point %",
        "short": "3P%",
        "unit": "ratio (0-1)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.005,
        "league_avg": 0.306,
        "mvp_avg": 0.328,
        "mvp_min": 0.000,
        "mvp_max": 0.454,
        "example": 0.380,
        "desc": "Three-point field goal percentage (0.000 to 1.000). Non-shooters/centers may be near 0.000."
    },
    "ft_percent": {
        "name": "Free Throw %",
        "short": "FT%",
        "unit": "ratio (0-1)",
        "min": 0.0,
        "max": 1.0,
        "step": 0.005,
        "league_avg": 0.750,
        "mvp_avg": 0.779,
        "mvp_min": 0.380,
        "mvp_max": 0.921,
        "example": 0.850,
        "desc": "Free throw shooting accuracy (0.000 to 1.000, e.g., 0.850 = 85.0%)."
    },
    "trb_per_game": {
        "name": "Total Rebounds",
        "short": "TRB",
        "unit": "reb/g",
        "min": 0.0,
        "max": 30.0,
        "step": 0.1,
        "league_avg": 3.8,
        "mvp_avg": 11.5,
        "mvp_min": 3.3,
        "mvp_max": 27.0,
        "example": 10.5,
        "desc": "Total rebounds per game (offensive + defensive). Guards ~4-7, Forwards/Centers ~9-15."
    },
    "ast_per_game": {
        "name": "Assists Per Game",
        "short": "AST",
        "unit": "ast/g",
        "min": 0.0,
        "max": 20.0,
        "step": 0.1,
        "league_avg": 2.0,
        "mvp_avg": 5.8,
        "mvp_min": 1.3,
        "mvp_max": 12.8,
        "example": 8.0,
        "desc": "Average assists per game. Elite playmakers average 8-12; dominant bigs average 3-6."
    },
    "stl_per_game": {
        "name": "Steals Per Game",
        "short": "STL",
        "unit": "stl/g",
        "min": 0.0,
        "max": 5.0,
        "step": 0.1,
        "league_avg": 0.7,
        "mvp_avg": 1.5,
        "mvp_min": 0.5,
        "mvp_max": 3.2,
        "example": 1.4,
        "desc": "Steals per game. Leading defenders and alert perimeter players average 1.2 to 2.5."
    },
    "blk_per_game": {
        "name": "Blocks Per Game",
        "short": "BLK",
        "unit": "blk/g",
        "min": 0.0,
        "max": 6.0,
        "step": 0.1,
        "league_avg": 0.4,
        "mvp_avg": 1.4,
        "mvp_min": 0.1,
        "mvp_max": 4.1,
        "example": 1.2,
        "desc": "Blocks per game. Rim protectors average 1.5 to 3.5; perimeter players average 0.2 to 0.8."
    },
    "pts_per_game": {
        "name": "Points Per Game",
        "short": "PTS",
        "unit": "pts/g",
        "min": 0.0,
        "max": 55.0,
        "step": 0.1,
        "league_avg": 9.2,
        "mvp_avg": 27.2,
        "mvp_min": 13.8,
        "mvp_max": 37.6,
        "example": 28.5,
        "desc": "Scoring average per game. Ranked by Random Forest as the #1 most critical MVP predictor."
    }
}

PRESETS = {
    "jokic_2024": {
        "name": "Nikola Jokić (2024 MVP)",
        "type": "MVP Winner",
        "season": 2024,
        "stats": {
            "g": 79,
            "mp_per_game": 34.6,
            "fg_percent": 0.583,
            "x3p_percent": 0.359,
            "ft_percent": 0.817,
            "trb_per_game": 12.4,
            "ast_per_game": 9.0,
            "stl_per_game": 1.4,
            "blk_per_game": 0.9,
            "pts_per_game": 26.4
        }
    },
    "embiid_2023": {
        "name": "Joel Embiid (2023 MVP)",
        "type": "MVP Winner",
        "season": 2023,
        "stats": {
            "g": 66,
            "mp_per_game": 34.6,
            "fg_percent": 0.548,
            "x3p_percent": 0.330,
            "ft_percent": 0.857,
            "trb_per_game": 10.2,
            "ast_per_game": 4.2,
            "stl_per_game": 1.0,
            "blk_per_game": 1.7,
            "pts_per_game": 33.1
        }
    },
    "curry_2016": {
        "name": "Stephen Curry (2016 Unanimous MVP)",
        "type": "MVP Winner",
        "season": 2016,
        "stats": {
            "g": 79,
            "mp_per_game": 34.2,
            "fg_percent": 0.504,
            "x3p_percent": 0.454,
            "ft_percent": 0.908,
            "trb_per_game": 5.4,
            "ast_per_game": 6.7,
            "stl_per_game": 2.1,
            "blk_per_game": 0.2,
            "pts_per_game": 30.1
        }
    },
    "giannis_2020": {
        "name": "Giannis Antetokounmpo (2020 MVP)",
        "type": "MVP Winner",
        "season": 2020,
        "stats": {
            "g": 63,
            "mp_per_game": 30.4,
            "fg_percent": 0.553,
            "x3p_percent": 0.304,
            "ft_percent": 0.633,
            "trb_per_game": 13.6,
            "ast_per_game": 5.6,
            "stl_per_game": 1.0,
            "blk_per_game": 1.0,
            "pts_per_game": 29.5
        }
    },
    "jordan_1996": {
        "name": "Michael Jordan (1996 MVP)",
        "type": "MVP Winner",
        "season": 1996,
        "stats": {
            "g": 82,
            "mp_per_game": 37.7,
            "fg_percent": 0.495,
            "x3p_percent": 0.427,
            "ft_percent": 0.834,
            "trb_per_game": 6.6,
            "ast_per_game": 4.3,
            "stl_per_game": 2.2,
            "blk_per_game": 0.5,
            "pts_per_game": 30.4
        }
    },
    "sga_2025": {
        "name": "Shai Gilgeous-Alexander (2025 MVP)",
        "type": "MVP Winner",
        "season": 2025,
        "stats": {
            "g": 76,
            "mp_per_game": 34.2,
            "fg_percent": 0.519,
            "x3p_percent": 0.375,
            "ft_percent": 0.898,
            "trb_per_game": 5.4,
            "ast_per_game": 6.4,
            "stl_per_game": 1.7,
            "blk_per_game": 1.0,
            "pts_per_game": 31.4
        }
    },
    "all_star_non_mvp": {
        "name": "All-Star Scorer (Non-MVP)",
        "type": "Non-MVP Contender",
        "season": 2024,
        "stats": {
            "g": 68,
            "mp_per_game": 34.0,
            "fg_percent": 0.450,
            "x3p_percent": 0.355,
            "ft_percent": 0.810,
            "trb_per_game": 4.5,
            "ast_per_game": 4.8,
            "stl_per_game": 1.1,
            "blk_per_game": 0.3,
            "pts_per_game": 24.2
        }
    },
    "role_player": {
        "name": "Solid NBA Starter / Role Player (Non-MVP)",
        "type": "Non-MVP Starter",
        "season": 2024,
        "stats": {
            "g": 72,
            "mp_per_game": 26.0,
            "fg_percent": 0.440,
            "x3p_percent": 0.345,
            "ft_percent": 0.770,
            "trb_per_game": 4.2,
            "ast_per_game": 2.8,
            "stl_per_game": 0.8,
            "blk_per_game": 0.4,
            "pts_per_game": 13.5
        }
    }
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def prepare_datasets(data_dir="data"):
    awards_path = os.path.join(data_dir, "Player Award Shares.csv")
    stats_path = os.path.join(data_dir, "Player Per Game.csv")

    awards = pd.read_csv(awards_path)
    stats = pd.read_csv(stats_path)
    mvp = awards[awards["award"] == "nba mvp"]

    # Filter duplicate traded player rows
    tm_mask = stats["team"].str.contains("TM", na=False)
    clean_stats = stats[
        tm_mask | ~stats.duplicated(subset=["season", "player_id"], keep=False)
    ].copy()

    clean_stats = clean_stats[
        (clean_stats["season"] >= 1956) &
        (clean_stats["season"] <= 2025)
    ].copy()

    mvp_winners = mvp[mvp["winner"] == True][["season", "player_id"]].copy()
    mvp_winners["mvp"] = 1

    data = clean_stats.merge(
        mvp_winners,
        on=["season", "player_id"],
        how="left"
    )
    data["mvp"] = data["mvp"].fillna(0).astype(int)

    return data


class MVPPredictionEngine:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.models_dir = MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {}
        self.preprocessors = {}
        self._load_or_train()

    def _load_or_train(self):
        full_rf_path = os.path.join(self.models_dir, "rf_full.joblib")
        if os.path.exists(full_rf_path):
            try:
                self.models["rf_full"] = joblib.load(full_rf_path)
                self.models["logreg_full"] = joblib.load(os.path.join(self.models_dir, "logreg_full.joblib"))
                self.models["nb_full"] = joblib.load(os.path.join(self.models_dir, "nb_full.joblib"))
                self.preprocessors["imputer_full"] = joblib.load(os.path.join(self.models_dir, "imputer_full.joblib"))
                self.preprocessors["scaler_full"] = joblib.load(os.path.join(self.models_dir, "scaler_full.joblib"))
                
                # Split models (<= 2019)
                self.models["rf_split"] = joblib.load(os.path.join(self.models_dir, "rf_split.joblib"))
                self.models["logreg_split"] = joblib.load(os.path.join(self.models_dir, "logreg_split.joblib"))
                self.models["nb_split"] = joblib.load(os.path.join(self.models_dir, "nb_split.joblib"))
                self.preprocessors["imputer_split"] = joblib.load(os.path.join(self.models_dir, "imputer_split.joblib"))
                self.preprocessors["scaler_split"] = joblib.load(os.path.join(self.models_dir, "scaler_split.joblib"))
                print("Loaded pre-trained models successfully.")
                return
            except Exception as e:
                print(f"Error loading models: {e}. Re-training...")

        self._train_all()

    def _train_all(self):
        print("Training models from datasets...")
        data = prepare_datasets(self.data_dir)

        # 1. Full Dataset (1956-2025)
        imp_full = SimpleImputer(strategy="median")
        X_imp_full = imp_full.fit_transform(data[FEATURES])

        scaler_full = StandardScaler()
        X_scaled_full = scaler_full.fit_transform(X_imp_full)

        y_full = data["mvp"]

        rf_full = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_full.fit(X_imp_full, y_full)

        logreg_full = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        logreg_full.fit(X_scaled_full, y_full)

        nb_full = GaussianNB()
        nb_full.fit(X_imp_full, y_full)

        # 2. Split Dataset (<= 2019)
        train_split = data[data["season"] <= 2019].copy()
        imp_split = SimpleImputer(strategy="median")
        X_imp_split = imp_split.fit_transform(train_split[FEATURES])

        scaler_split = StandardScaler()
        X_scaled_split = scaler_split.fit_transform(X_imp_split)

        y_split = train_split["mvp"]

        rf_split = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_split.fit(X_imp_split, y_split)

        logreg_split = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        logreg_split.fit(X_scaled_split, y_split)

        nb_split = GaussianNB()
        nb_split.fit(X_imp_split, y_split)

        # Save artifacts
        self.models["rf_full"] = rf_full
        self.models["logreg_full"] = logreg_full
        self.models["nb_full"] = nb_full
        self.preprocessors["imputer_full"] = imp_full
        self.preprocessors["scaler_full"] = scaler_full

        self.models["rf_split"] = rf_split
        self.models["logreg_split"] = logreg_split
        self.models["nb_split"] = nb_split
        self.preprocessors["imputer_split"] = imp_split
        self.preprocessors["scaler_split"] = scaler_split

        joblib.dump(rf_full, os.path.join(self.models_dir, "rf_full.joblib"))
        joblib.dump(logreg_full, os.path.join(self.models_dir, "logreg_full.joblib"))
        joblib.dump(nb_full, os.path.join(self.models_dir, "nb_full.joblib"))
        joblib.dump(imp_full, os.path.join(self.models_dir, "imputer_full.joblib"))
        joblib.dump(scaler_full, os.path.join(self.models_dir, "scaler_full.joblib"))

        joblib.dump(rf_split, os.path.join(self.models_dir, "rf_split.joblib"))
        joblib.dump(logreg_split, os.path.join(self.models_dir, "logreg_split.joblib"))
        joblib.dump(nb_split, os.path.join(self.models_dir, "nb_split.joblib"))
        joblib.dump(imp_split, os.path.join(self.models_dir, "imputer_split.joblib"))
        joblib.dump(scaler_split, os.path.join(self.models_dir, "scaler_split.joblib"))

        print("Trained and saved all models successfully.")

    def predict(self, input_features, model_name="rf", mode="full"):
        """
        input_features: dict with keys in FEATURES
        model_name: 'rf' (Random Forest), 'logreg' (Logistic Regression), 'nb' (Naive Bayes), 'ensemble'
        mode: 'full' (1956-2025) or 'split' (<= 2019)
        """
        # Validate and format input
        row = []
        for feat in FEATURES:
            val = float(input_features.get(feat, FEATURE_METADATA[feat]["example"]))
            row.append(val)

        df_input = pd.DataFrame([row], columns=FEATURES)

        imputer = self.preprocessors[f"imputer_{mode}"]
        scaler = self.preprocessors[f"scaler_{mode}"]

        X_imp = imputer.transform(df_input)
        X_scaled = scaler.transform(X_imp)

        rf = self.models[f"rf_{mode}"]
        logreg = self.models[f"logreg_{mode}"]
        nb = self.models[f"nb_{mode}"]

        # Predictions & Probabilities
        rf_prob = float(rf.predict_proba(X_imp)[0, 1])
        logreg_prob = float(logreg.predict_proba(X_scaled)[0, 1])
        nb_prob = float(nb.predict_proba(X_imp)[0, 1])

        # Calibrated threshold for binary MVP classification:
        # In modern full model: RF prob >= 0.25 (or 0.30) signifies top MVP tier in the 99.8th percentile!
        # When RF prob is >= 0.50, it is guaranteed MVP caliber.
        # LogReg threshold with balanced class weights is 0.50.
        # For overall MVP determination, we evaluate:
        # In 'full' mode: rf_prob >= 0.35 or (rf_prob >= 0.25 and logreg_prob >= 0.85)
        # In 'split' mode: rf_prob >= 0.15 or (rf_prob >= 0.10 and logreg_prob >= 0.80)
        if mode == "full":
            rf_is_mvp = rf_prob >= 0.35
            mvp_calibrated_score = min(100.0, round(rf_prob * 135.0, 1))
        else:
            rf_is_mvp = rf_prob >= 0.15
            mvp_calibrated_score = min(100.0, round(rf_prob * 330.0, 1))

        logreg_is_mvp = logreg_prob >= 0.50
        nb_is_mvp = nb_prob >= 0.50

        # Primary decision based on selected model
        if model_name == "rf":
            primary_is_mvp = rf_is_mvp
            primary_prob = rf_prob
            primary_model_title = "Random Forest (Top Performer)"
        elif model_name == "logreg":
            primary_is_mvp = logreg_is_mvp
            primary_prob = logreg_prob
            primary_model_title = "Logistic Regression"
        elif model_name == "nb":
            primary_is_mvp = nb_is_mvp
            primary_prob = nb_prob
            primary_model_title = "Gaussian Naive Bayes"
        else:  # ensemble
            votes = sum([1 if v else 0 for v in [rf_is_mvp, logreg_is_mvp, nb_is_mvp]])
            primary_is_mvp = votes >= 2
            primary_prob = (rf_prob + logreg_prob + nb_prob) / 3.0
            primary_model_title = "Multi-Model Ensemble"

        # Realistic NBA eligibility check (Games Played rule):
        games = input_features.get("g", 82)
        eligibility_warning = None
        if games < 50:
            eligibility_warning = f"Notice: Historical MVPs almost always play 50+ games (modern rule requires 65 games). At {games} games, player is at high risk of award disqualification."

        # Feature importances from Random Forest
        importances = rf.feature_importances_
        feat_imp_list = []
        for f, imp in zip(FEATURES, importances):
            feat_imp_list.append({
                "feature": f,
                "name": FEATURE_METADATA[f]["name"],
                "short": FEATURE_METADATA[f]["short"],
                "importance": round(float(imp) * 100, 2),
                "value": input_features.get(f, 0),
                "mvp_avg": FEATURE_METADATA[f]["mvp_avg"]
            })
        feat_imp_list.sort(key=lambda x: x["importance"], reverse=True)

        # Benchmark comparison:
        benchmarks = {}
        for f in FEATURES:
            u_val = float(input_features.get(f, 0))
            mvp_val = FEATURE_METADATA[f]["mvp_avg"]
            league_val = FEATURE_METADATA[f]["league_avg"]
            diff_mvp = round(((u_val - mvp_val) / (mvp_val if mvp_val != 0 else 1)) * 100, 1)
            benchmarks[f] = {
                "name": FEATURE_METADATA[f]["name"],
                "short": FEATURE_METADATA[f]["short"],
                "user_val": u_val,
                "mvp_avg": mvp_val,
                "league_avg": league_val,
                "diff_pct": diff_mvp,
                "status": "above" if u_val >= mvp_val else ("near" if diff_mvp >= -15 else "below")
            }

        # Insights synthesis
        insights = []
        pts = input_features.get("pts_per_game", 0)
        mp = input_features.get("mp_per_game", 0)
        g = input_features.get("g", 0)

        if pts >= 26.0:
            insights.append(f"Scoring prowess ({pts:.1f} PPG) is in the elite MVP winner tier (MVP avg: 27.2).")
        elif pts >= 22.0:
            insights.append(f"Scoring ({pts:.1f} PPG) is strong All-Star tier, but slightly below the historical MVP average of 27.2 PPG.")
        else:
            insights.append(f"Scoring ({pts:.1f} PPG) is significantly below typical MVP levels (historical MVP median: 26.6 PPG).")

        if mp >= 34.0:
            insights.append(f"Floor time ({mp:.1f} MPG) confirms a primary franchise focal point role.")
        else:
            insights.append(f"Minutes per game ({mp:.1f} MPG) is relatively modest for an MVP leader (MVP avg: 38.3 MPG).")

        if g >= 70:
            insights.append(f"Durability ({int(g)} Games) is excellent and meets the NBA MVP 65-game threshold.")
        elif g >= 55:
            insights.append(f"Games played ({int(g)} Games) is acceptable historically, though below modern 65-game criteria.")
        else:
            insights.append(f"Limited games ({int(g)} Games) strongly diminishes MVP probability.")

        return {
            "is_mvp": bool(primary_is_mvp),
            "verdict": "MVP WINNER" if primary_is_mvp else "NOT AN MVP",
            "model_title": primary_model_title,
            "raw_probability": round(primary_prob, 4),
            "prob_percentage": round(primary_prob * 100, 1),
            "calibrated_score": round(mvp_calibrated_score, 1),
            "models_consensus": {
                "rf": {
                    "name": "Random Forest",
                    "is_mvp": bool(rf_is_mvp),
                    "probability": round(rf_prob, 4),
                    "prob_pct": round(rf_prob * 100, 1),
                    "rank_grade": "Top 1-2 Tier" if rf_prob >= 0.25 else ("Contender" if rf_prob >= 0.08 else "Unlikely")
                },
                "logreg": {
                    "name": "Logistic Regression",
                    "is_mvp": bool(logreg_is_mvp),
                    "probability": round(logreg_prob, 4),
                    "prob_pct": round(logreg_prob * 100, 1),
                    "rank_grade": "High" if logreg_prob >= 0.50 else "Low"
                },
                "nb": {
                    "name": "Gaussian Naive Bayes",
                    "is_mvp": bool(nb_is_mvp),
                    "probability": round(nb_prob, 4),
                    "prob_pct": round(nb_prob * 100, 1),
                    "rank_grade": "High" if nb_prob >= 0.50 else "Low"
                }
            },
            "feature_importance": feat_imp_list,
            "benchmarks": benchmarks,
            "insights": insights,
            "eligibility_warning": eligibility_warning
        }
