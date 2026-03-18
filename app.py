import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, classification_report
)

#config page
st.set_page_config(
    page_title="Spaceship Titanic – ML Prédictif",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
sns.set_theme(style="darkgrid", palette="muted")

# chargement et processing
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("train.csv")
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['IsChild']       = (df['Age'] < 13).astype(int)
    df['Group_Id']      = df['PassengerId'].str.split('_').str[0]
    df['Group_Size']    = df.groupby('Group_Id')['Group_Id'].transform('count')

    # CryoSleep = True → dépenses = 0
    for col in spending_cols:
        df.loc[df['CryoSleep'] == True, col] = 0

    # HomePlanet manquante → mode du groupe
    df['HomePlanet'] = df.groupby('Group_Id')['HomePlanet'].transform(
        lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
    )

    # Extraction Cabin
    df['Deck']     = df['Cabin'].apply(lambda x: ord(x.split('/')[0].upper()) - ord('A') if pd.notna(x) else np.nan)
    df['CabinNum'] = df['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) else np.nan)
    df['CabinSide']= df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else np.nan)

    # Imputation
    num_cols  = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
    df[num_cols]             = df[num_cols].fillna(0)
    df[['VIP', 'CryoSleep']] = df[['VIP', 'CryoSleep']].fillna(False)

    # Encodage
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'CabinSide'])

    # Scaling
    scaler       = StandardScaler()
    num_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'CabinNum']
    df[num_features] = scaler.fit_transform(df[num_features])
    df['TotalSpending'] = df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].sum(axis=1)

    X = df.drop(columns=['Transported', 'PassengerId', 'Name', 'Cabin', 'Group_Id'], errors='ignore')
    y = df['Transported'].astype(int)

    return df, X, y, scaler

@st.cache_resource
def train_model(_X, _y):
    if os.path.exists("models/best_model.joblib"):
        return joblib.load("models/best_model.joblib")

    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42, stratify=_y)
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators':      [100, 200],
        'max_depth':         [5, 10, None],
        'min_samples_split': [2, 5],
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_estimator_, "models/best_model.joblib")
    return grid.best_estimator_

# Chargement
try:
    df_raw              = pd.read_csv("train.csv")
    df, X, y, scaler    = load_and_preprocess()
    model               = train_model(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred              = model.predict(X_test)
    y_proba             = model.predict_proba(X_test)[:, 1]
    acc                 = accuracy_score(y_test, y_pred)
    auc                 = roc_auc_score(y_test, y_proba)
    data_ok             = True
except FileNotFoundError:
    data_ok = False

#sidebar
st.sidebar.title("🚀 Spaceship Titanic")
menu = [
    " Accueil",
    " Analyse Exploratoire (EDA)",
    " Test du Modèle",
    " Évaluation des Performances",
    " Conclusions & Perspectives"
]
choice = st.sidebar.radio("Navigation :", menu)
st.sidebar.markdown("---")

if data_ok:
    st.sidebar.success(f"✅ Dataset : **{df_raw.shape[0]} passagers**")
    st.sidebar.info(f"🎯 Accuracy : **{acc:.2%}**  |  AUC : **{auc:.4f}**")
else:
    st.sidebar.error("❌ `train.csv` introuvable.\nPlace-le dans le même dossier que `app.py`.")

st.sidebar.markdown("---")
st.sidebar.caption("TP Machine Learning – Spaceship Titanic")

#page d'accuel
if choice == " Accueil":
    st.title("🚀 Spaceship Titanic – Machine Learning Prédictif")
    st.markdown("""
    > **Objectif :** Concevoir un système expert capable de déterminer la probabilité
    > de transport d'un passager vers une **dimension alternative** en fonction de son profil.
    """)

    if data_ok:
        transported = int(df_raw['Transported'].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Total passagers", f"{df_raw.shape[0]:,}")
        c2.metric("✨ Transportés",      f"{transported:,}")
        c3.metric("🚀 Non transportés",  f"{df_raw.shape[0] - transported:,}")
        c4.metric("🎯 Accuracy modèle",  f"{acc:.2%}")

    st.markdown("---")

    with st.expander("📋 Aperçu du dataset brut", expanded=True):
        if data_ok:
            st.dataframe(df_raw.head(10), use_container_width=True)

    with st.expander("📌 Structure du projet"):
        st.code("""
spaceship_titanic/
├── app.py               ← Application Streamlit
├── train.csv            ← Dataset Kaggle
├── test.csv
├── models/
│   └── best_model.joblib
└── requirements.txt
        """, language="markdown")

#page2 EDA
elif choice == " Analyse Exploratoire (EDA)":
    st.title("📊 Analyse Exploratoire des Données")

    if not data_ok:
        st.error("train.csv introuvable.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Variable Cible",
        "📊 Variables Catégorielles",
        "📈 Variables Numériques",
        "👥 Groupes & Dépenses"
    ])

    # Tab 1 Pie chart
    with tab1:
        st.subheader("Répartition de la variable Transported")
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(5, 5))
            counts = df_raw["Transported"].value_counts()
            ax.pie(counts,
                   labels=["Non transporté", "Transporté"],
                   autopct="%1.1f%%",
                   colors=["#E07B54", "#5DA5C8"],
                   startangle=90,
                   wedgeprops=dict(edgecolor="white", linewidth=2))
            ax.set_title("Transported", fontweight="bold")
            st.pyplot(fig)
        with col2:
            st.markdown("### 📌 Observations")
            st.markdown("""
            - Dataset **quasi équilibré** : ~50.3% transportés vs 49.7% non transportés
            - Pas de déséquilibre de classes → pas besoin de SMOTE
            - L'accuracy est une métrique fiable dans ce cas
            """)
            st.dataframe(
                df_raw["Transported"].value_counts()
                .rename("Nombre").reset_index()
                .rename(columns={"Transported": "Statut"}),
                use_container_width=True
            )

    #Tab 2 stacked bar
    with tab2:
        st.subheader("Taux de transport par variable catégorielle")
        cat_var = st.selectbox("Choisir une variable :", ["HomePlanet", "CryoSleep", "Destination", "VIP"])

        fig, ax = plt.subplots(figsize=(8, 4))
        ct = pd.crosstab(df_raw[cat_var], df_raw["Transported"], normalize="index") * 100
        ct.columns = ["Non transporté", "Transporté"]
        ct.plot(kind="bar", stacked=True, ax=ax, color=["#E07B54", "#5DA5C8"], edgecolor="white")
        ax.set_title(f"Taux de transport par {cat_var}", fontweight="bold")
        ax.set_ylabel("Pourcentage (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Influence du Pont (Deck) et du Côté (Side)")
        df_eda = df_raw.copy()
        df_eda[["Deck_raw", "Num_raw", "Side_raw"]] = df_eda["Cabin"].str.split("/", expand=True)

        col1, col2 = st.columns(2)
        for col_name, col_widget in [("Deck_raw", col1), ("Side_raw", col2)]:
            with col_widget:
                fig, ax = plt.subplots(figsize=(6, 4))
                ct2 = pd.crosstab(df_eda[col_name], df_eda["Transported"], normalize="index") * 100
                ct2.columns = ["Non transporté", "Transporté"]
                ct2.plot(kind="bar", stacked=True, ax=ax, color=["#E07B54", "#5DA5C8"], edgecolor="white")
                ax.set_title(f"Par {col_name.replace('_raw','')}", fontweight="bold")
                ax.set_ylabel("%")
                ax.tick_params(axis="x", rotation=0)
                st.pyplot(fig)

    # Tab 3 numérique
    with tab3:
        st.subheader("Distribution de l'Âge (KDE)")
        fig, ax = plt.subplots(figsize=(9, 4))
        df_raw[df_raw["Transported"] == True]["Age"].dropna().plot.kde(
            ax=ax, label="Transporté", linewidth=2, color="#5DA5C8")
        df_raw[df_raw["Transported"] == False]["Age"].dropna().plot.kde(
            ax=ax, label="Non transporté", linewidth=2, color="#E07B54")
        ax.set_title("Distribution de l'âge selon le statut de transport", fontweight="bold")
        ax.set_xlabel("Âge")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        col1, col2 = st.columns(2)
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        with col1:
            st.subheader("Heatmap de corrélation – Dépenses")
            fig, ax = plt.subplots(figsize=(6, 5))
            mask = np.triu(np.ones_like(df_raw[spending_cols].corr(), dtype=bool))
            sns.heatmap(df_raw[spending_cols].corr(), annot=True, fmt=".2f",
                        cmap="coolwarm", mask=mask, linewidths=0.5, ax=ax)
            ax.set_title("Pearson – Postes de dépenses", fontweight="bold")
            st.pyplot(fig)

        with col2:
            st.subheader("Boxplot – Dépenses totales")
            df_raw["TotalSpending_viz"] = df_raw[spending_cols].sum(axis=1)
            fig, ax = plt.subplots(figsize=(6, 5))
            df_raw.boxplot(column="TotalSpending_viz", by="Transported", ax=ax)
            ax.set_title("Dépenses totales vs Transported", fontweight="bold")
            ax.set_xlabel("Transported")
            ax.set_ylabel("Total Spending ($)")
            plt.suptitle("")
            st.pyplot(fig)

    # Tab 4 : Groupes
    with tab4:
        st.subheader("Survie selon la taille du groupe")
        df_g = df_raw.copy()
        df_g["Group_Id"]   = df_g["PassengerId"].str.split("_").str[0]
        df_g["Group_Size"] = df_g.groupby("Group_Id")["Group_Id"].transform("count")

        fig, ax = plt.subplots(figsize=(10, 4))
        pd.crosstab(df_g["Group_Size"], df_g["Transported"], normalize="index").plot(
            kind="bar", stacked=True, ax=ax, color=["#E07B54", "#5DA5C8"], edgecolor="white"
        )
        ax.set_xlabel("Taille du groupe")
        ax.set_ylabel("Proportion")
        ax.set_title("Taux de transport selon la taille du groupe", fontweight="bold")
        ax.legend(["Non transporté", "Transporté"])
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

        st.dataframe(
            df_g.groupby("Group_Size")["Transported"]
            .mean().sort_values(ascending=False)
            .rename("Taux de transport")
            .reset_index(),
            use_container_width=True
        )

#Page 3 test du modèle
elif choice == " Test du Modèle":
    st.title("🤖 Calculateur de Survie")
    st.markdown("Remplis ton profil passager et le modèle prédit si tu seras **transporté dans une dimension alternative** !")

    if not data_ok:
        st.error("train.csv introuvable.")
        st.stop()

    with st.form("prediction_form"):
        st.subheader("👤 Profil passager")

        col1, col2, col3 = st.columns(3)
        with col1:
            age         = st.number_input("Âge", min_value=0, max_value=100, value=25)
            home_planet = st.selectbox("Planète d'origine", ["Earth", "Europa", "Mars"])
            cryo_sleep  = st.selectbox("CryoSleep ?", [False, True])
        with col2:
            destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
            vip         = st.selectbox("VIP ?", [False, True])
            cabin_side  = st.selectbox("Côté cabine (Side)", ["S", "P"])
        with col3:
            deck_letter = st.selectbox("Pont (Deck)", ["A","B","C","D","E","F","G","T"])
            cabin_num   = st.number_input("Numéro de cabine", min_value=0, max_value=2000, value=100)

        st.markdown("---")
        st.subheader("💰 Dépenses à bord")
        c1, c2, c3, c4, c5 = st.columns(5)
        room_service  = c1.number_input("RoomService",  min_value=0, value=0)
        food_court    = c2.number_input("FoodCourt",    min_value=0, value=0)
        shopping_mall = c3.number_input("ShoppingMall", min_value=0, value=0)
        spa           = c4.number_input("Spa",          min_value=0, value=0)
        vr_deck       = c5.number_input("VRDeck",       min_value=0, value=0)

        submitted = st.form_submit_button("🚀 Lancer la prédiction", use_container_width=True)

    if submitted:
        with st.spinner("Analyse en cours..."):

            # Si CryoSleep → dépenses = 0
            if cryo_sleep:
                room_service = food_court = shopping_mall = spa = vr_deck = 0

            # Scaling des variables numériques (même ordre que l'entraînement)
            num_raw    = np.array([[age, room_service, food_court, shopping_mall, spa, vr_deck, cabin_num]], dtype=float)
            num_scaled = scaler.transform(num_raw)[0]
            age_s, rs_s, fc_s, sm_s, spa_s, vrd_s, cn_s = num_scaled
            total_s    = rs_s + fc_s + sm_s + spa_s + vrd_s

            # Construction du vecteur (même colonnes que X d'entraînement)
            input_dict = {col: 0 for col in X.columns}

            # Numériques
            input_dict['Age']          = age_s
            input_dict['RoomService']  = rs_s
            input_dict['FoodCourt']    = fc_s
            input_dict['ShoppingMall'] = sm_s
            input_dict['Spa']          = spa_s
            input_dict['VRDeck']       = vrd_s
            input_dict['CabinNum']     = cn_s
            input_dict['TotalSpending']= total_s
            input_dict['IsChild']      = 1 if age < 13 else 0
            input_dict['Group_Size']   = 1
            input_dict['Deck']         = ord(deck_letter.upper()) - ord('A')
            input_dict['CryoSleep']    = int(cryo_sleep)
            input_dict['VIP']          = int(vip)

            # One-hot HomePlanet
            for planet in ["Earth", "Europa", "Mars"]:
                col_name = f"HomePlanet_{planet}"
                if col_name in input_dict:
                    input_dict[col_name] = 1 if home_planet == planet else 0

            # One-hot Destination
            for dest in ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"]:
                col_name = f"Destination_{dest}"
                if col_name in input_dict:
                    input_dict[col_name] = 1 if destination == dest else 0

            # One-hot CabinSide
            for side in ["S", "P"]:
                col_name = f"CabinSide_{side}"
                if col_name in input_dict:
                    input_dict[col_name] = 1 if cabin_side == side else 0

            input_df   = pd.DataFrame([input_dict])
            prediction = model.predict(input_df)[0]
            proba      = model.predict_proba(input_df)[0]

        # Résultat
        st.markdown("---")
        st.subheader("🎯 Résultat")

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success(f"## ✨ Vous serez TRANSPORTÉ !")
            else:
                st.warning(f"## 🚀 Vous ne serez PAS transporté.")
            st.markdown(f"### Probabilité de transport : **{proba[1]*100:.1f}%**")
            st.progress(float(proba[1]))

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["Non transporté", "Transporté"], proba * 100,
                   color=["#E07B54", "#5DA5C8"], edgecolor="white", linewidth=1.5)
            ax.set_ylabel("Probabilité (%)")
            ax.set_title("Confiance du modèle", fontweight="bold")
            ax.set_ylim(0, 100)
            for i, v in enumerate(proba * 100):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
            st.pyplot(fig)

#Page 4 évaluation des performances
elif choice == " Évaluation des Performances":
    st.title("📈 Évaluation des Performances du Modèle")

    if not data_ok:
        st.error("train.csv introuvable.")
        st.stop()

    # Métriques clés
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Accuracy",  f"{acc:.2%}")
    c2.metric("📊 AUC-ROC",   f"{auc:.4f}")
    c3.metric("📋 F1-Score",  f"{float(classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']):.2%}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matrice de Confusion")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Non transporté", "Transporté"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title("Matrice de Confusion", fontweight="bold")
        st.pyplot(fig)

    with col2:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="#5DA5C8", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.fill_between(fpr, tpr, alpha=0.1, color="#5DA5C8")
        ax.set_xlabel("Taux de faux positifs (FPR)")
        ax.set_ylabel("Taux de vrais positifs (TPR)")
        ax.set_title("Courbe ROC", fontweight="bold")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Feature Importances – Top 15")
    feature_names = X.columns.tolist()
    importances   = model.feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=fi_df, y="Feature", x="Importance", palette="mako", ax=ax)
    ax.set_title("Top 15 Feature Importances – Random Forest", fontweight="bold")
    ax.set_xlabel("Importance (Mean Decrease Impurity)")
    st.pyplot(fig)

    top3 = fi_df["Feature"].head(3).tolist()
    st.markdown("### 🔍 Les 3 variables qui déterminent le plus le destin d'un passager :")
    for i, feat in enumerate(top3, 1):
        imp = fi_df.loc[fi_df["Feature"] == feat, "Importance"].values[0]
        st.markdown(f"**{i}. {feat}** — importance = `{imp:.4f}`")

    st.markdown("---")
    st.subheader("Rapport de classification complet")
    report = classification_report(y_test, y_pred, target_names=["Non transporté", "Transporté"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

#Page 5 conclusions et perspectives
elif choice == " Conclusions & Perspectives":
    st.title("💡 Conclusions & Pistes d'Amélioration")

    st.markdown("### 📝 Bilan du Projet")
    if data_ok:
        st.markdown(f"""
        Le modèle **Random Forest** entraîné sur le dataset Spaceship Titanic obtient :
        - **Accuracy** : `{acc:.2%}`
        - **AUC-ROC**  : `{auc:.4f}`

        Ces performances montrent que le modèle est capable de discriminer efficacement
        les passagers transportés des non-transportés.
        """)

    st.markdown("### 🔍 Facteurs clés identifiés")
    st.markdown("""
    D'après l'analyse des feature importances du Random Forest, les variables
    les plus déterminantes sont **CryoSleep**, **TotalSpending** et **Deck**.
    - Les passagers en **CryoSleep** ont un taux de transport nettement plus élevé
    - Les passagers ayant de **faibles dépenses** sont plus souvent transportés
    - Certains **ponts du vaisseau** présentent des taux de transport très différents
    """)

    st.markdown("### 🚧 Pistes d'amélioration")
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("✨ Ajouter XGBoost / LightGBM pour comparer")
        st.checkbox("🔁 Implémenter une Cross-Validation plus robuste (StratifiedKFold)")
        st.checkbox("📊 Ajouter les SHAP values pour l'explicabilité")
    with col2:
        st.checkbox("🧹 Feature engineering plus poussé (interactions entre variables)")
        st.checkbox("☁️ Déploiement sur Streamlit Cloud ou HuggingFace Spaces")
        st.checkbox("⚡ Optimisation avec RandomizedSearchCV sur plus d'hyperparamètres")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>TP Machine Learning – Spaceship Titanic 🚀</p>",
    unsafe_allow_html=True
)