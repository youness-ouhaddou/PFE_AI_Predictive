import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════
# 1. CHARGEMENT ET STANDARDISATION DU STOCK (F1)
# ══════════════════════════════════════════════════
df_stock = pd.read_excel('data/stock_pdr.xlsx', header=0)
# Standardisation des colonnes comme dans le script de nettoyage
df_stock.columns = ['Article', 'Designation', 'Plant', 'Emplacement', 'Qte_stock', 'Montant_conso', 'Prix_F1']
df_stock['Article'] = df_stock['Article'].astype(str).str.strip()
df_stock = df_stock[['Article', 'Designation', 'Qte_stock', 'Prix_F1']].drop_duplicates('Article')
df_stock['Prix_F1'] = df_stock['Prix_F1'].replace(0, np.nan)

# ══════════════════════════════════════════════════
# 2. EXTRACTION DE LA MATRICE DE CONSO PIVOT (F2)
# ══════════════════════════════════════════════════
# Chargement sans en-tête car le fichier contient 4 lignes de métadonnées
df_conso_raw = pd.read_excel('data/consommations.xlsx', header=None)

row0 = df_conso_raw.iloc[0].tolist()   # Types de mouvement SAP (101, 102, 201, 202)
row1 = df_conso_raw.iloc[1].tolist()   # Liste des Mois (01.2024, 02.2024, ...)
row2 = df_conso_raw.iloc[2].tolist()   # REC Qty ou Iss. Qty

# Repérage des colonnes de sorties réelles (201) et annulations (202)
cols_201 = [i for i, (mt, mois, ri) in enumerate(zip(row0, row1, row2))
            if str(mt) == '201' and 'Iss' in str(ri) and mois != 'Result']
cols_202 = [i for i, (mt, mois, ri) in enumerate(zip(row0, row1, row2))
            if str(mt) == '202' and 'Iss' in str(ri) and mois != 'Result']

# Définition propre et dynamique de la liste des mois
mois_cols = [str(row1[i]) for i in cols_201]

# Extraction et nettoyage des lignes articles
data_conso = df_conso_raw.iloc[4:].copy()
data_conso['Article'] = data_conso[2].astype(str).str.strip()
data_conso = data_conso[data_conso['Article'].notna() & (~data_conso['Article'].isin(['nan', 'None']))].copy()

# Calcul des consommations mensuelles brutes (201)
mat_201 = data_conso[cols_201].apply(pd.to_numeric, errors='coerce').fillna(0)
mat_201.index = data_conso['Article'].values
mat_201.columns = mois_cols

# Alignement et soustraction des retours/annulations (202)
mat_202_aligned = pd.DataFrame(0, index=data_conso['Article'].values, columns=mois_cols)
for col_202 in cols_202:
    m_202 = row1[col_202]
    if m_202 in mois_cols:
        vals = data_conso[col_202].apply(pd.to_numeric, errors='coerce').fillna(0).values
        mat_202_aligned[m_202] += vals

# Consommation nette finale (Loi de Pareto / IA)
mat_conso_nette = (mat_201 - mat_202_aligned.abs()).clip(lower=0)

# ══════════════════════════════════════════════════
# 3. INGENIERIE DES FEATURES POUR L'IA (ANTI-LEAKAGE)
# ══════════════════════════════════════════════════
df_features = pd.DataFrame(index=mat_conso_nette.index)

# CIBLE : Le mois le plus récent (Mois N)
df_features['Conso_Futur'] = mat_conso_nette[mois_cols[-1]]

# HISTORIQUE : S'arrête STRICTEMENT au mois N-1 pour éliminer la triche (Data Leakage)
cols_historique = mois_cols[:-1]
df_features['Conso_Mois_N']  = mat_conso_nette[mois_cols[-2]] # Mois précédant immédiatement la cible
df_features['Conso_Mois_N1'] = mat_conso_nette[mois_cols[-3]] # Mois N-2
df_features['Conso_Mois_N2'] = mat_conso_nette[mois_cols[-4]] # Mois N-3

# Statistiques calculées exclusivement sur le passé de référence stable
df_features['Conso_Moyenne'] = mat_conso_nette[cols_historique].mean(axis=1)
df_features['Conso_Ecart']   = mat_conso_nette[cols_historique].std(axis=1).fillna(0)

# ══════════════════════════════════════════════════
# 4. RECONSTITUTION DU DATASET ET PARETO ABC
# ══════════════════════════════════════════════════
df_features = df_features.reset_index().rename(columns={'index': 'Article'})
df_merged = df_stock.merge(df_features, on='Article', how='inner')
# Injection des colonnes temporelles mensuelles complètes dans df_merged pour l'étape suivante
df_merged = df_merged.merge(mat_conso_nette.reset_index().rename(columns={'index': 'Article'}), on='Article', how='inner')

# Calcul de la valeur consommée pour appliquer la loi de Pareto
df_merged['Prix_unitaire'] = df_merged['Prix_F1'].fillna(0)
df_merged['Valeur_consommee'] = df_merged['Conso_Moyenne'] * df_merged['Prix_unitaire'] * 12

df_merged = df_merged.sort_values(by='Valeur_consommee', ascending=False).reset_index(drop=True)
df_merged['Cumul_pct'] = (df_merged['Valeur_consommee'].cumsum() / df_merged['Valeur_consommee'].sum() * 100).fillna(100)

df_merged['Classe_ABC'] = df_merged['Cumul_pct'].apply(lambda pct: 'A' if pct <= 80 else ('B' if pct <= 95 else 'C'))

print(f"✅ Données réelles chargées avec succès : {len(df_merged)} articles appairés.")
print(f"   Période historique d'apprentissage : {mois_cols[0]} ➔ {mois_cols[-2]}")
print(f"   Mois cible pour l'évaluation IA   : {mois_cols[-1]}")
print(f"   Articles avec consommation active  : {(df_merged['Conso_Moyenne'] > 0).sum()}")

# ══════════════════════════════════════════════════
# VÉRIFICATION ADÉQUATION LOI DE POISSON
# ══════════════════════════════════════════════════
print("\n╔══════════════════════════════════════════╗")
print("║  VÉRIFICATION LOI DE POISSON — PDR       ║")
print("╚══════════════════════════════════════════╝")

for classe in ['A', 'B', 'C']:
    mask = df_merged['Classe_ABC'] == classe
    subset = df_merged[mask][mois_cols]
    
    if len(subset) == 0:
        continue
        
    lambda_reel = subset.values.mean()
    sigma_reel  = subset.values.std()
    cv_reel     = sigma_reel / lambda_reel if lambda_reel > 0 else 0
    
    print(f"\n  Classe {classe} ({len(subset)} articles) :")
    print(f"    λ — Moyenne mensuelle  : {lambda_reel:.3f} pièces/mois")
    print(f"    σ — Écart-type         : {sigma_reel:.3f}")
    print(f"    CV (σ/μ)               : {cv_reel:.3f}")
    print(f"    Poisson valide ?       : "
          f"{'✅ CV≈1 → Poisson adapté' if 0.7 < cv_reel < 1.35 else '⚠️ CV hors plage (Demande erratique)'}")
    
    # ══════════════════════════════════════════════════
# SAUVEGARDE DU DATASET POUR L'ENTRAÎNEMENT DE L'IA
# ══════════════════════════════════════════════════
# On ne garde que les colonnes nécessaires à l'IA pour l'apprentissage
colonnes_ia = [
    'Article', 'Designation', 'Classe_ABC', 'Qte_stock', 'Prix_unitaire',
    'Conso_Mois_N', 'Conso_Mois_N1', 'Conso_Mois_N2', 
    'Conso_Moyenne', 'Conso_Ecart', 'Conso_Futur'
]

df_ia_final = df_merged[colonnes_ia]

# Sauvegarde en format CSV dans ton dossier data
df_ia_final.to_csv('data/dataset_pdr_ia.csv', index=False, encoding='utf-8')

print("\n💾 [SUCCESS] Le dataset d'apprentissage a été sauvegardé !")
print("   ➔ Fichier créé : 'data/dataset_pdr_ia.csv'")
print("   ➔ Prêt pour l'entraînement des modèles de Régression et Random Forest.")