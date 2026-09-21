"""
IPL Match Winner Prediction
============================
Backend  : Scikit-learn (Random Forest Classifier)
Frontend : Streamlit + Plotly + Matplotlib + Seaborn
Dataset  : IPL_Matches_Data_2008_2026.csv
"""

import os
import warnings
import joblib
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "IPL_Matches_Data_2008_2026.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ipl_rf_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "ipl_encoders.pkl")

FEATURES = ["team1", "team2", "toss_winner", "toss_decision", "city"]
TARGET = "winner"

TEAM_COLORS = {
    "Chennai Super Kings": "#F5A623",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bangalore": "#C8102E",
    "Kolkata Knight Riders": "#552583",
    "Rajasthan Royals": "#EA1A7F",
    "Sunrisers Hyderabad": "#FF822A",
    "Delhi Capitals": "#00008B",
    "Punjab Kings": "#AA4069",
    "Lucknow Super Giants": "#00B4D8",
    "Gujarat Titans": "#1B6CA8",
    "Deccan Chargers": "#4A90D9",
    "Kochi Tuskers Kerala": "#2ECC71",
    "Pune Warriors": "#8E44AD",
    "Rising Pune Supergiant": "#E74C3C",
    "Rising Pune Supergiants": "#E74C3C",
}

# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & preprocessing dataset …")
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    # Keep only completed matches with a winner
    df = df[df["result_type"] == "complete"].copy()
    df = df.dropna(subset=FEATURES + [TARGET])

    # Normalise team names (strip whitespace)
    for col in ["team1", "team2", "toss_winner", TARGET]:
        df[col] = df[col].str.strip()

    # Derive toss advantage flag
    df["toss_advantage"] = (df["toss_winner"] == df[TARGET]).astype(int)

    # Parse year from date or season
    df["year"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.year
    df["year"] = df["year"].fillna(
        df["season"].astype(str).str[:4].astype(float)
    ).astype("Int64")

    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING & ENCODING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model …")
def train_model(df: pd.DataFrame):
    encoders = {}  # type: Dict[str, LabelEncoder]
    df_enc = df[FEATURES + [TARGET]].copy()

    cat_cols = ["team1", "team2", "toss_winner", "toss_decision", "city"]
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    df_enc[TARGET] = le_target.fit_transform(df_enc[TARGET].astype(str))
    encoders[TARGET] = le_target

    X = df_enc[FEATURES]
    y = df_enc[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save model + encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    return model, encoders, X_test, y_test, y_pred, acc, report, cm, X_train


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_winner(model, encoders, team1, team2, toss_winner, toss_decision, city):
    try:
        row = {
            "team1": encoders["team1"].transform([team1])[0],
            "team2": encoders["team2"].transform([team2])[0],
            "toss_winner": encoders["toss_winner"].transform([toss_winner])[0],
            "toss_decision": encoders["toss_decision"].transform([toss_decision])[0],
            "city": encoders["city"].transform([city])[0],
        }
    except ValueError as e:
        return None, None, str(e)

    X_input = pd.DataFrame([row])
    proba = model.predict_proba(X_input)[0]
    pred_idx = np.argmax(proba)
    pred_team = encoders[TARGET].inverse_transform([pred_idx])[0]
    classes = encoders[TARGET].inverse_transform(range(len(proba)))
    proba_dict = dict(zip(classes, proba))
    return pred_team, proba_dict, None


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="IPL Match Predictor",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.6rem;
            font-weight: 800;
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #555;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: white;
            text-align: center;
        }
        .winner-box {
            background: linear-gradient(135deg, #f5a623, #f76c1c);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            color: white;
            font-size: 1.8rem;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="main-header">🏏 IPL Match Winner Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Powered by Random Forest · Dataset: 2008 – 2026 · Built with Streamlit</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load data & train ──────────────────────
    df = load_data()
    model, encoders, X_test, y_test, y_pred, acc, report, cm, X_train = train_model(df)

    all_teams = sorted(df["team1"].unique().tolist())
    all_cities = sorted(df["city"].dropna().unique().tolist())

    # ── Sidebar ────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/IPL_Logo_2022.svg/300px-IPL_Logo_2022.svg.png",
            use_column_width=True,
        )
        st.header("⚙️ Navigation")
        page = st.radio(
            "Go to",
            ["🔮 Predict Match", "📊 Data Insights", "🤖 Model Performance", "📋 Dataset Explorer"],
        )
        st.divider()
        st.markdown(f"**Total Matches:** {len(df):,}")
        st.markdown(f"**Seasons:** 2008 – 2026")
        st.markdown(f"**Teams:** {df['team1'].nunique()}")
        st.markdown(f"**Model Accuracy:** `{acc:.2%}`")

    # ══════════════════════════════════════════
    # PAGE 1 — PREDICT MATCH
    # ══════════════════════════════════════════
    if page == "🔮 Predict Match":
        st.subheader("🔮 Predict the Winner of an IPL Match")
        st.write("Fill in the pre-match details below and click **Predict**.")

        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("🔵 Team 1", all_teams, index=0)
            toss_winner = st.selectbox("🪙 Toss Winner", [team1, "Other"], index=0)
        with col2:
            team2_opts = [t for t in all_teams if t != team1]
            team2 = st.selectbox("🔴 Team 2", team2_opts, index=0)
            toss_decision = st.radio("🏃 Toss Decision", ["bat", "field"], horizontal=True)

        city = st.selectbox("🏙️ City / Venue City", all_cities, index=0)

        # Resolve toss_winner
        if toss_winner == "Other":
            toss_winner = team2

        if st.button("⚡ Predict Winner", use_container_width=True, type="primary"):
            if team1 == team2:
                st.error("Team 1 and Team 2 cannot be the same!")
            else:
                pred_team, proba_dict, err = predict_winner(
                    model, encoders, team1, team2, toss_winner, toss_decision, city
                )
                if err:
                    st.warning(f"Prediction error: {err}")
                else:
                    st.success("Prediction Complete!")
                    res_col1, res_col2 = st.columns([1, 2])

                    with res_col1:
                        color = TEAM_COLORS.get(pred_team, "#667eea")
                        st.markdown(
                            f'<div class="winner-box" style="background:{color};">🏆 {pred_team}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("Predicted Winner")

                    with res_col2:
                        # Show top-5 probabilities
                        top_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:6]
                        teams_plot = [t for t, _ in top_proba]
                        probs_plot = [p * 100 for _, p in top_proba]
                        colors_bar = [TEAM_COLORS.get(t, "#aaa") for t in teams_plot]

                        fig = go.Figure(
                            go.Bar(
                                x=probs_plot,
                                y=teams_plot,
                                orientation="h",
                                marker_color=colors_bar,
                                text=[f"{p:.1f}%" for p in probs_plot],
                                textposition="outside",
                            )
                        )
                        fig.update_layout(
                            title="Win Probability (%)",
                            xaxis_title="Probability (%)",
                            height=280,
                            margin=dict(l=10, r=10, t=40, b=10),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # ── Head-to-Head History ────────────────
        st.divider()
        st.subheader("📜 Head-to-Head History")
        if "team1" in locals() and "team2" in locals():
            h2h = df[
                ((df["team1"] == team1) & (df["team2"] == team2)) |
                ((df["team1"] == team2) & (df["team2"] == team1))
            ]
            t1_wins = (h2h["winner"] == team1).sum()
            t2_wins = (h2h["winner"] == team2).sum()
            total_h2h = len(h2h)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"🏆 {team1} Wins", t1_wins)
            c2.metric("Total Matches", total_h2h)
            c3.metric(f"🏆 {team2} Wins", t2_wins)

            if total_h2h > 0:
                fig_pie = px.pie(
                    names=[team1, team2],
                    values=[t1_wins, t2_wins],
                    color_discrete_sequence=[
                        TEAM_COLORS.get(team1, "#4A90D9"),
                        TEAM_COLORS.get(team2, "#E74C3C"),
                    ],
                    title=f"H2H Win Split: {team1} vs {team2}",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    # ══════════════════════════════════════════
    # PAGE 2 — DATA INSIGHTS
    # ══════════════════════════════════════════
    elif page == "📊 Data Insights":
        st.subheader("📊 IPL Data Insights (2008 – 2026)")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🏆 Most Wins", "🪙 Toss Analysis", "🏙️ Venue Stats", "📅 Season Trend"]
        )

        with tab1:
            wins = df["winner"].value_counts().reset_index()
            wins.columns = ["Team", "Wins"]
            wins["Color"] = wins["Team"].map(lambda t: TEAM_COLORS.get(t, "#aaa"))
            fig = px.bar(
                wins.head(12),
                x="Wins",
                y="Team",
                orientation="h",
                color="Team",
                color_discrete_map={r["Team"]: r["Color"] for _, r in wins.iterrows()},
                title="Most IPL Match Wins (All Time)",
            )
            fig.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            toss_df = df.copy()
            toss_df["toss_win_match_win"] = toss_df["toss_winner"] == toss_df["winner"]
            toss_pct = toss_df.groupby("toss_decision")["toss_win_match_win"].mean().reset_index()
            toss_pct.columns = ["Decision", "Win Rate"]
            toss_pct["Win Rate %"] = (toss_pct["Win Rate"] * 100).round(2)

            fig_toss = px.bar(
                toss_pct,
                x="Decision",
                y="Win Rate %",
                color="Decision",
                title="Toss Winner Win Rate by Decision",
                text="Win Rate %",
                color_discrete_sequence=["#F5A623", "#004BA0"],
            )
            fig_toss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_toss, use_container_width=True)

            st.info(
                f"Overall: Toss winner wins **{toss_df['toss_win_match_win'].mean():.1%}** of matches."
            )

        with tab3:
            venue_wins = df.groupby("city")["winner"].count().reset_index()
            venue_wins.columns = ["City", "Matches"]
            venue_wins = venue_wins.sort_values("Matches", ascending=False).head(15)
            fig_v = px.bar(
                venue_wins,
                x="City",
                y="Matches",
                color="Matches",
                color_continuous_scale="Blues",
                title="Top 15 Host Cities by Number of Matches",
            )
            fig_v.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_v, use_container_width=True)

        with tab4:
            season_df = df.dropna(subset=["year"])
            season_matches = season_df.groupby("year").size().reset_index(name="Matches")
            fig_s = px.line(
                season_matches,
                x="year",
                y="Matches",
                markers=True,
                title="Number of Matches Per Season",
                labels={"year": "Season", "Matches": "Matches Played"},
            )
            fig_s.update_traces(line_color="#667eea", marker_color="#f5a623")
            st.plotly_chart(fig_s, use_container_width=True)

    # ══════════════════════════════════════════
    # PAGE 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════
    elif page == "🤖 Model Performance":
        st.subheader("🤖 Random Forest Classifier — Model Evaluation")

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{acc:.2%}")
        col2.metric("Training Samples", f"{len(X_train):,}")
        col3.metric("Test Samples", f"{len(X_test):,}")

        st.divider()

        # Feature Importance
        st.markdown("### 📌 Feature Importances")
        fi = pd.DataFrame(
            {"Feature": FEATURES, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            fi,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Feature Importance (Random Forest)",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Confusion Matrix (top 10 teams only for readability)
        st.markdown("### 🔢 Confusion Matrix (Top 10 Teams)")
        le_target = encoders[TARGET]
        top_teams = df["winner"].value_counts().head(10).index.tolist()
        top_idx = [i for i, t in enumerate(le_target.classes_) if t in top_teams]

        mask_test = np.isin(y_test, top_idx)
        y_test_filt = y_test[mask_test]
        y_pred_filt = np.array(y_pred)[mask_test]

        cm_filt = confusion_matrix(y_test_filt, y_pred_filt, labels=top_idx)
        top_labels = [le_target.classes_[i] for i in top_idx]

        fig_cm, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            cm_filt,
            annot=True,
            fmt="d",
            xticklabels=top_labels,
            yticklabels=top_labels,
            cmap="YlOrRd",
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix — Top 10 Teams")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_cm)

        # Classification Report
        st.markdown("### 📄 Classification Report (Top 10 Teams)")
        report_df = (
            pd.DataFrame(report)
            .transpose()
            .drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
            .reset_index()
            .rename(columns={"index": "Team"})
        )
        report_df = report_df[report_df["Team"].isin(top_teams)]
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True)

    # ══════════════════════════════════════════
    # PAGE 4 — DATASET EXPLORER
    # ══════════════════════════════════════════
    elif page == "📋 Dataset Explorer":
        st.subheader("📋 Raw Dataset Explorer")
        st.write(f"Showing {len(df):,} completed IPL matches (2008 – 2026)")

        season_filter = st.multiselect(
            "Filter by Season",
            sorted(df["year"].dropna().unique().tolist(), reverse=True),
            default=[],
        )
        team_filter = st.multiselect("Filter by Team (any role)", all_teams, default=[])

        view_df = df.copy()
        if season_filter:
            view_df = view_df[view_df["year"].isin(season_filter)]
        if team_filter:
            view_df = view_df[
                view_df["team1"].isin(team_filter) | view_df["team2"].isin(team_filter)
            ]

        display_cols = [
            "date", "city", "team1", "team2", "toss_winner", "toss_decision",
            "winner", "result_type", "win_by_runs", "win_by_wickets", "player_of_match",
        ]
        st.dataframe(
            view_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=500,
        )
        st.caption(f"Filtered rows: {len(view_df):,}")

    # Footer
    st.divider()
    st.markdown(
        "<center><sub>🏏 IPL Match Predictor · Built with Streamlit & Scikit-learn · Data: 2008–2026</sub></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
