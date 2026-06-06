import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Créer le dossier s'il n'existe pas
os.makedirs('outputs/graphs', exist_ok=True)

# Charger tous les modèles
model_if = joblib.load('models/isolation_forest_model.pkl')
scaler   = joblib.load('models/scaler_if.pkl')

# Charger le dataset
df = pd.read_csv('outputs/dataset_variateur.csv')

print("✅ Tous les modèles chargés")
print("✅ Dataset chargé")
print("\n⏳ Calcul des métriques en cours...")

from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, roc_curve)
import seaborn as sns

# ─── Prédiction à la volée ───
features = ['Temperature', 'Courant', 'Tension', 'RunHours', 'FaultCount']
X_scaled = scaler.transform(df[features])

predictions = model_if.predict(X_scaled)
scores = model_if.score_samples(X_scaled)

# ─── Labels vrais et prédits ───
# normal = 0 | degradation + panne = 1 (anomalie)
y_true = (df['Label'] != 'normal').astype(int)
y_pred = (predictions == -1).astype(int) 
scores_roc = -scores # Inversion des scores pour la courbe ROC

# ─── Calcul des métriques ───
cm        = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)
accuracy  = (TP + TN) / len(y_true)
auc_score = roc_auc_score(y_true, scores_roc)
fpr, tpr, _ = roc_curve(y_true, scores_roc)

print("╔══════════════════════════════════════════╗")
print("║  MÉTRIQUES — ISOLATION FOREST            ║")
print("╚══════════════════════════════════════════╝")
print(f"\n  Matrice de confusion :")
print(f"    TN (normaux corrects)    : {TN}")
print(f"    FP (fausses alarmes)     : {FP}")
print(f"    FN (pannes manquées ⚠️)  : {FN}")
print(f"    TP (anomalies détectées) : {TP}")

print(f"\n  Précision   : {precision:.3f}  → {precision*100:.1f}% des alarmes levées sont réelles")
print(f"  Rappel      : {recall:.3f}  → {recall*100:.1f}% des vraies anomalies détectées")
print(f"  Score F1    : {f1:.3f}  → Équilibre Précision/Rappel")
print(f"  Accuracy    : {accuracy:.3f}  → {accuracy*100:.1f}% de classifications correctes")
print(f"  ROC-AUC     : {auc_score:.3f}  → {'Excellent' if auc_score > 0.9 else 'Bon' if auc_score > 0.8 else 'Acceptable'}")

print(f"\n  Priorité Rappel > Précision :")
print(f"  En maintenance, manquer une panne (FN) est")
print(f"  plus coûteux qu'une fausse alarme (FP)")

# ─── Graphiques des métriques IF ───
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Métriques de validation — Isolation Forest', fontsize=13, fontweight='bold')

# Matrice de confusion
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomalie'],
            yticklabels=['Normal', 'Anomalie'],
            ax=ax1, linewidths=0.5, annot_kws={'size': 14})
ax1.set_xlabel('Prédit', fontsize=11)
ax1.set_ylabel('Réel', fontsize=11)
ax1.set_title(f'Matrice de Confusion\nPrécision={precision:.2f} | Rappel={recall:.2f} | F1={f1:.2f}', fontsize=10)

# Courbe ROC
ax2 = axes[1]
ax2.plot(fpr, tpr, color='#2ecc71', linewidth=2.5, label=f'Isolation Forest (AUC = {auc_score:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Classifieur aléatoire (AUC = 0.5)')
ax2.fill_between(fpr, tpr, alpha=0.15, color='#2ecc71')
ax2.set_xlabel('Taux Faux Positifs (1 − Spécificité)', fontsize=11)
ax2.set_ylabel('Taux Vrais Positifs (Rappel)', fontsize=11)
ax2.set_title(f'Courbe ROC\nAUC = {auc_score:.3f}', fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('outputs/graphs/metriques_isolation_forest.png', dpi=150, bbox_inches='tight')
print("\n✅ metriques_isolation_forest.png sauvegardé")
plt.show()

# ─── Calcul des métriques ───
cm        = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)
accuracy  = (TP + TN) / len(y_true)

scores_roc = -df['Score_IF']
auc_score  = roc_auc_score(y_true, scores_roc)
fpr, tpr, _ = roc_curve(y_true, scores_roc)

print("╔══════════════════════════════════════════╗")
print("║  MÉTRIQUES — ISOLATION FOREST            ║")
print("╚══════════════════════════════════════════╝")
print(f"\n  Matrice de confusion :")
print(f"    TN (normaux corrects)    : {TN}")
print(f"    FP (fausses alarmes)     : {FP}")
print(f"    FN (pannes manquées ⚠️)  : {FN}")
print(f"    TP (anomalies détectées) : {TP}")
print(f"\n  Précision   : {precision:.3f}  → {precision*100:.1f}% "
      f"des alarmes levées sont réelles")
print(f"  Rappel      : {recall:.3f}  → {recall*100:.1f}% "
      f"des vraies anomalies détectées")
print(f"  Score F1    : {f1:.3f}  → Équilibre Précision/Rappel")
print(f"  Accuracy    : {accuracy:.3f}  → {accuracy*100:.1f}% "
      f"de classifications correctes")
print(f"  ROC-AUC     : {auc_score:.3f}  → "
      f"{'Excellent' if auc_score > 0.9 else 'Bon' if auc_score > 0.8 else 'Acceptable'}")
print(f"\n  Priorité Rappel > Précision :")
print(f"  En maintenance, manquer une panne (FN) est")
print(f"  plus coûteux qu'une fausse alarme (FP)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Métriques de validation — Isolation Forest',
             fontsize=13, fontweight='bold')

# ─── Matrice de confusion ───
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomalie'],
            yticklabels=['Normal', 'Anomalie'],
            ax=ax1, linewidths=0.5, annot_kws={'size': 14})
ax1.set_xlabel('Prédit', fontsize=11)
ax1.set_ylabel('Réel', fontsize=11)
ax1.set_title(f'Matrice de Confusion\nPrécision={precision:.2f} | '
              f'Rappel={recall:.2f} | F1={f1:.2f}', fontsize=10)

# ─── Courbe ROC ───
ax2 = axes[1]
ax2.plot(fpr, tpr, color='#2ecc71', linewidth=2.5,
         label=f'Isolation Forest (AUC = {auc_score:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1,
         alpha=0.5, label='Classifieur aléatoire (AUC = 0.5)')
ax2.fill_between(fpr, tpr, alpha=0.15, color='#2ecc71')
ax2.set_xlabel('Taux Faux Positifs (1 − Spécificité)', fontsize=11)
ax2.set_ylabel('Taux Vrais Positifs (Rappel)', fontsize=11)
ax2.set_title(f'Courbe ROC\nAUC = {auc_score:.3f}', fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('outputs/graphs/metriques_isolation_forest.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ metriques_isolation_forest.png sauvegardé")