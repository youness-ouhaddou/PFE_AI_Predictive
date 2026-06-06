from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

df = pd.read_csv('outputs/dataset_variateur.csv')
print(f"✅ Dataset chargé : {len(df)} lignes")
# ─── Features utilisées ───
features = ['Temperature', 'Courant', 'Tension', 'RunHours', 'FaultCount']
X = df[features].copy()

# ─── Normalisation ───
# StandardScaler : μ=0, σ=1 pour chaque feature
# Nécessaire car Temperature (~45) >> FaultCount (~0-7)
# Sans normalisation, Temperature dominerait le modèle
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Entraînement sur données normales uniquement ───
X_normal = X_scaled[df['Label'] == 'normal']

model_if = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model_if.fit(X_normal)

print("✅ Modèle Isolation Forest entraîné")
print(f"   Nombre d'arbres : {model_if.n_estimators}")
print(f"   Features        : {features}")
print(f"   Données train   : {len(X_normal)} points normaux")

# ─── Prédiction sur toutes les données ───
predictions       = model_if.predict(X_scaled)
scores            = model_if.score_samples(X_scaled)
df['Anomalie_IF'] = (predictions == -1).astype(int)
df['Score_IF']    = scores

# ─── Résultats par scénario ───
print("\n📊 RÉSULTATS PAR SCÉNARIO :")
for label in ['normal', 'degradation', 'panne']:
    subset = df[df['Label'] == label]
    taux   = subset['Anomalie_IF'].mean() * 100
    print(f"   {label:14s} : {taux:5.1f}% détectés comme anomalies")

# ─── Sauvegarde des modèles ───
joblib.dump(model_if, 'models/isolation_forest_model.pkl')
joblib.dump(scaler,   'models/scaler_if.pkl')
print("\n✅ Modèles sauvegardés dans models/")

import joblib
joblib.dump(model_if, 'models/isolation_forest_model.pkl')
joblib.dump(scaler,   'models/scaler_if.pkl')
print("✅ Modèles sauvegardés :")
print("   models/isolation_forest_model.pkl")
print("   models/scaler_if.pkl")