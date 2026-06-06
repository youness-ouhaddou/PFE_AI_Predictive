import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)  # Reproductibilité garantie

# ══════════════════════════════════════════════════
# GÉNÉRATION BASÉE SUR FC_SIMULATOR (TIA PORTAL V17)
# ══════════════════════════════════════════════════

# ─── SCÉNARIO 0 : Normal (300 points) ───
# Source : FC_Simulator Scénario 0
# Sim_Temp    := 45.0 + SIN(Sim_TimeCounter) * 3.0
# Sim_Current := 11.5 + SIN(Sim_TimeCounter) * 0.5
t_norm = np.linspace(0, 50, 300)
temp_normal    = 45.0 + np.sin(t_norm) * 3.0 + np.random.normal(0, 0.5, 300)
current_normal = 11.5 + np.sin(t_norm) * 0.5 + np.random.normal(0, 0.2, 300)
tension_normal = np.full(300, 400.0) + np.random.normal(0, 1.0, 300)
hours_normal   = np.linspace(0, 3000, 300)
faults_normal  = np.zeros(300)

# ─── SCÉNARIO 1 : Dégradation (150 points) ───
# Source : FC_Simulator Scénario 1
# Sim_Temp := 45.0 + (Sim_TimeCounter * 0.55) → 45°C à 78°C sur 60s
t_deg = np.linspace(0, 60, 150)
temp_deg    = 45.0 + t_deg * 0.55 + np.random.normal(0, 0.8, 150)
current_deg = 11.5 + t_deg * 0.10 + np.random.normal(0, 0.3, 150)
tension_deg = 400.0 - t_deg * 0.05 + np.random.normal(0, 1.0, 150)
hours_deg   = np.linspace(3000, 6000, 150)
faults_deg  = np.random.choice([0, 1, 2], 150, p=[0.7, 0.2, 0.1])

# ─── SCÉNARIO 2 : Panne (50 points) ───
# Source : FC_Simulator Scénario 2
# Sim_Temp := 80.0 (seuil disjonction SINAMICS S120)
temp_panne    = np.full(50, 80.0) + np.random.normal(0, 0.5, 50)
current_panne = np.zeros(50) + np.random.normal(0, 0.1, 50)
tension_panne = np.zeros(50)
hours_panne   = np.linspace(6000, 8000, 50)
faults_panne  = np.random.randint(3, 8, 50)

# ─── ASSEMBLAGE ───
df = pd.DataFrame({
    'Temperature': np.concatenate([temp_normal, temp_deg, temp_panne]),
    'Courant':     np.concatenate([current_normal, current_deg, current_panne]),
    'Tension':     np.concatenate([tension_normal, tension_deg, tension_panne]),
    'RunHours':    np.concatenate([hours_normal, hours_deg, hours_panne]),
    'FaultCount':  np.concatenate([faults_normal, faults_deg, faults_panne]),
    'Label':       ['normal']*300 + ['degradation']*150 + ['panne']*50
})

df.to_csv('outputs/dataset_variateur.csv', index=False)
print(f"✅ Dataset généré : {len(df)} points")
print(df.groupby('Label').size())

# ══════════════════════════════════════════════════════════════
# ANALYSE STATISTIQUE COMPLÈTE — RÉPONSE À L'ENCADRANT
# ══════════════════════════════════════════════════════════════

print("╔══════════════════════════════════════════════════╗")
print("║  ANALYSE STATISTIQUE DES DONNÉES GÉNÉRÉES       ║")
print("╚══════════════════════════════════════════════════╝")

for scenario in ['normal', 'degradation', 'panne']:
    subset = df[df['Label'] == scenario]
    print(f"\n{'═'*50}")
    print(f"SCÉNARIO : {scenario.upper()} (n={len(subset)})")
    print(f"{'═'*50}")

    for col in ['Temperature', 'Courant', 'Tension']:
        valeurs = subset[col]
        q1, q3  = valeurs.quantile(0.25), valeurs.quantile(0.75)
        # Sécurité pour éviter la division par zéro sur le CV
        moyenne = valeurs.mean()
        ecart_type = valeurs.std()
    
        if moyenne == 0:
            cv = float('nan')
        else:
            cv = ecart_type / moyenne * 100
        
        # Sécurité pour le test de Shapiro (impossible si la donnée est constante)
        if ecart_type == 0:
            stat, p = 1.0, 1.0  # Valeur par défaut si donnée constante
        else:
            stat, p = stats.shapiro(valeurs)

            print(f"\n  {col} :")
            print(f"    Moyenne (μ)       : {valeurs.mean():.3f}")
            print(f"    Écart-type (σ)    : {valeurs.std():.3f}")
            print(f"    Médiane           : {valeurs.median():.3f}")
            print(f"    Min / Max         : {valeurs.min():.3f} / {valeurs.max():.3f}")
            print(f"    Q1 / Q3           : {q1:.3f} / {q3:.3f}")
            print(f"    IQR               : {q3 - q1:.3f}")
            print(f"    CV (σ/μ × 100)    : {cv:.1f}%")

        # Test de normalité Shapiro-Wilk
        sample      = valeurs.sample(min(50, len(valeurs)), random_state=42)
        stat, p     = stats.shapiro(sample)
        normalite   = "✅ Normale (p>0.05)" if p > 0.05 else "⚠️ Non-normale"
        print(f"    Shapiro-Wilk      : stat={stat:.4f}, p={p:.4f} → {normalite}")

# ─── Comparaison source automate vs données générées ───
print(f"\n{'═'*50}")
print("COMPARAISON : AUTOMATE TIA PORTAL vs DONNÉES GÉNÉRÉES")
print(f"{'═'*50}")

normal_data = df[df['Label'] == 'normal']
print(f"\n  Température : μ_automate=45.0°C | "
      f"μ_généré={normal_data['Temperature'].mean():.2f}°C | "
      f"Écart={abs(45.0 - normal_data['Temperature'].mean()):.3f}°C")
print(f"  Courant     : μ_automate=11.5A  | "
      f"μ_généré={normal_data['Courant'].mean():.2f}A  | "
      f"Écart={abs(11.5 - normal_data['Courant'].mean()):.3f}A")

# ─── Cohérence avec seuils FC_RiskScore ───
print(f"\n{'═'*50}")
print("COHÉRENCE AVEC SEUILS FC_RISKSCORE (FC1)")
print(f"{'═'*50}")
seuils = {
    'Watch    (T > 60°C)': (df['Temperature'] > 60).sum(),
    'Critique (T > 75°C)': (df['Temperature'] > 75).sum(),
    'Surint.  (I > 18A) ': (df['Courant'] > 18).sum(),
}
for label, count in seuils.items():
    pct = count / len(df) * 100
    print(f"  {label} : {count:4d} pts ({pct:.1f}%) "
          f"→ {'✅ Cohérent' if pct < 40 else '⚠️ Vérifier'}")
    


# ══════════════════════════════════════════════════════════════
# GRAPHIQUE : Distributions statistiques par scénario
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    'Justification statistique des données générées\n'
    'Source : FC_Simulator (TIA Portal V17) + Standards Siemens SINAMICS S120',
    fontsize=12, fontweight='bold'
)

colors    = {'normal': '#2ecc71', 'degradation': '#f39c12', 'panne': '#e74c3c'}
variables = ['Temperature', 'Courant', 'Tension']
labels_fr = ['Température (°C)', 'Courant (A)', 'Tension (V)']

# Ligne 1 : Histogrammes avec courbe normale
for j, (var, label) in enumerate(zip(variables, labels_fr)):
    ax = axes[0][j]
    for scenario, color in colors.items():
        subset = df[df['Label'] == scenario][var]
        ax.hist(subset, bins=20, alpha=0.6, color=color,
                label=f"{scenario} (μ={subset.mean():.1f}, σ={subset.std():.1f})",
                density=True, edgecolor='none')
        mu, sigma = subset.mean(), subset.std()
        if sigma > 0:
            x_range = np.linspace(subset.min(), subset.max(), 100)
            ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                    color=color, linewidth=2, linestyle='--')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Densité', fontsize=10)
    ax.set_title(f'Distribution — {label}', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Ligne 2 : Boxplots avec annotations μ et σ
for j, (var, label) in enumerate(zip(variables, labels_fr)):
    ax = axes[1][j]
    data_by_scenario = [
        df[df['Label'] == s][var].values
        for s in ['normal', 'degradation', 'panne']
    ]
    bp = ax.boxplot(data_by_scenario, patch_artist=True,
                    tick_labels=['Normal', 'Dégradation', 'Panne'])
    for patch, color in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for k, data in enumerate(data_by_scenario):
        ax.annotate(
            f'μ={np.mean(data):.1f}\nσ={np.std(data):.1f}',
            xy=(k+1, np.percentile(data, 75)),
            ha='center', va='bottom', fontsize=8,
            color=list(colors.values())[k], fontweight='bold'
        )
    ax.set_xlabel('Scénario', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Boxplot — {label}', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/graphs/justification_statistique.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Graphique justification_statistique.png sauvegardé")


df.to_csv('outputs/dataset_variateur.csv', index=False)
print(f"✅ Dataset sauvegardé : outputs/dataset_variateur.csv")
print(f"   Taille : {len(df)} lignes × {len(df.columns)} colonnes")