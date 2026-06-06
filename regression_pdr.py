import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ─── CONFIGURATION DES CHEMINS (Identiques à votre script de nettoyage) ───
FICHIER_F1 = 'data/stock_pdr.xlsx'
FICHIER_F2 = 'data/consommations.xlsx'
FICHIER_F3 = 'data/commandes.xlsx'

print("=" * 65)
print("  ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION PDR — SMURFIT WESTROCK")
print("=" * 65)

# ─── ESSAYER DE CHARGER ET PARSER LES VRAIES DONNÉES S004 ───
try:
    print("\n[1/4] Lecture et parsing des fichiers Excel réels...")
    
    # 1. Chargement et nettoyage de F1 (Stock)
    f1 = pd.read_excel(FICHIER_F1, header=0)
    f1.columns = ['Article', 'Designation', 'Plant', 'Emplacement', 'Qte_stock', 'Montant_conso', 'Prix_F1']
    f1['Article'] = f1['Article'].astype(str).str.strip()
    f1 = f1[['Article', 'Qte_stock', 'Prix_F1']].drop_duplicates('Article')
    f1['Prix_F1'] = f1['Prix_F1'].replace(0, np.nan)

    # 2. Chargement et décodage du Pivot F2 (Consommations)
    df2_raw = pd.read_excel(FICHIER_F2, header=None)
    row0 = df2_raw.iloc[0].tolist()   # Types de mouvement (201, 202...)
    row1 = df2_raw.iloc[1].tolist()   # Mois (01.2024, 02.2024...)
    row2 = df2_raw.iloc[2].tolist()   # REC Qty / Iss. Qty
    
    cols_201 = [i for i, (mt, mois, ri) in enumerate(zip(row0, row1, row2))
                if str(mt) == '201' and 'Iss' in str(ri) and mois != 'Result']
    cols_202 = [i for i, (mt, mois, ri) in enumerate(zip(row0, row1, row2))
                if str(mt) == '202' and 'Iss' in str(ri) and mois != 'Result']
    mois_201 = [row1[i] for i in cols_201]

    data = df2_raw.iloc[4:].copy()
    data['Article'] = data[2].astype(str).str.strip()
    data = data[data['Article'].notna() & (~data['Article'].isin(['nan', 'None']))].copy()

    # Reconstruction des matrices de consommation mensuelle nette
    mat_201 = data[cols_201].apply(pd.to_numeric, errors='coerce').fillna(0)
    mat_201.index = data['Article'].values
    mat_201.columns = mois_201

    mat_202_aligned = pd.DataFrame(0, index=data['Article'].values, columns=mois_201)
    for col_202 in cols_202:
        m_202 = row1[col_202]
        if m_202 in mois_201:
            mat_202_aligned[m_202] += data[col_202].apply(pd.to_numeric, errors='coerce').fillna(0).values

    mat_conso_nette = (mat_201 - mat_202_aligned.abs()).clip(lower=0)
    mat_conso_nette = mat_conso_nette.groupby(mat_conso_nette.index).sum() # Unicité des articles

    # 3. Chargement de F3 (Commandes) pour les délais de livraison
    f3 = pd.read_excel(FICHIER_F3, header=0)
    f3.columns = ['Fournisseur_code', 'Fournisseur_nom', 'Commande_num', 'Article', 'Designation_F3', 
                  'Date_demande', 'Date_reception', 'Qte_demandee', 'Valeur_demandee', 'Qte_livree', 
                  'Valeur_livree', 'Qte_facturee', 'Montant_facture', 'Prix_F3', 'Article_maint']
    f3['Article'] = f3['Article'].astype(str).str.strip()
    f3 = f3[~f3['Article'].isin(['#', 'nan', 'None'])]
    f3['Date_demande'] = pd.to_datetime(f3['Date_demande'], dayfirst=True, errors='coerce')
    f3['Date_reception'] = pd.to_datetime(f3['Date_reception'], dayfirst=True, errors='coerce')
    f3['Delai_jours'] = (f3['Date_reception'] - f3['Date_demande']).dt.days
    
    f3_agg = f3.groupby('Article').agg(Delai_Fournisseur=('Delai_jours', 'mean'), Prix_F3=('Prix_F3', 'mean')).reset_index()

    # 4. Fusion globale et calcul des features statiques (ABC Code)
    f2_total = pd.DataFrame({'Article': mat_conso_nette.index, 'Conso_nette': mat_conso_nette.sum(axis=1).values})
    df_static = f1.merge(f2_total, on='Article', how='left').merge(f3_agg, on='Article', how='left')
    df_static['Conso_nette'] = df_static['Conso_nette'].fillna(0)
    df_static['Prix_unitaire'] = np.where(df_static['Prix_F1'].notna() & (df_static['Prix_F1'] > 0), df_static['Prix_F1'], df_static['Prix_F3'])
    df_static['Valeur_consommee'] = df_static['Conso_nette'] * df_static['Prix_unitaire'].fillna(0)
    
    # Classification ABC réglementaire
    df_actifs = df_static[df_static['Valeur_consommee'] > 0].sort_values('Valeur_consommee', ascending=False).copy()
    if len(df_actifs) > 0:
        df_actifs['Cumul_pct'] = (df_actifs['Valeur_consommee'].cumsum() / df_actifs['Valeur_consommee'].sum() * 100)
        df_actifs['Classe_ABC'] = df_actifs['Cumul_pct'].apply(lambda x: 'A' if x <= 80 else ('B' if x <= 95 else 'C'))
        df_static = df_static.merge(df_actifs[['Article', 'Classe_ABC']], on='Article', how='left')
    else:
        df_static['Classe_ABC'] = 'C'
    df_static['Classe_ABC'] = df_static['Classe_ABC'].fillna('C')
    df_static['ABC_Code'] = df_static['Classe_ABC'].map({'A': 3, 'B': 2, 'C': 1})
    
    # Fallbacks pour valeurs vides
    df_static['Delai_Fournisseur'] = df_static['Delai_Fournisseur'].fillna(f3['Delai_jours'].median()).round(1)
    
    # Simulation AMDEC (car non encore saisie par le service maintenance sur le terrain)
    np.random.seed(42)
    df_static['IPR_AMDEC'] = np.random.choice([8, 18, 36, 48], len(df_static))

    print("✅ Données réelles S004 chargées et nettoyées avec succès.")

    # ─── CRÉATION DES FENÊTRES GLISSANTES POUR LE MACHINE LEARNING ───
    print("\n[2/4] Génération de la matrice d'apprentissage (Sliding Window)...")
    records = []
    # On parcourt les 27 mois pour extraire des séquences de 4 mois consécutifs
    for i in range(len(mois_201) - 3):
        m_n2, m_n1, m_n, m_futur = mois_201[i], mois_201[i+1], mois_201[i+2], mois_201[i+3]
        
        for art in mat_conso_nette.index:
            art_info = df_static[df_static['Article'] == art]
            if art_info.empty: continue
            
            conso_n2 = mat_conso_nette.loc[art, m_n2]
            conso_n1 = mat_conso_nette.loc[art, m_n1]
            conso_n  = mat_conso_nette.loc[art, m_n]
            conso_futur = mat_conso_nette.loc[art, m_futur]
            
            # Filtre pour éviter d'entraîner le modèle sur des milliers de lignes de pièces jamais utilisées (0, 0, 0 -> 0)
            if (conso_n2 > 0) or (conso_n1 > 0) or (conso_n > 0) or (conso_futur > 0):
                records.append({
                    'Conso_Mois_N2': conso_n2,
                    'Conso_Mois_N1': conso_n1,
                    'Conso_Mois_N':  conso_n,
                    'IPR_AMDEC':     art_info['IPR_AMDEC'].values[0],
                    'Delai_Fournisseur': art_info['Delai_Fournisseur'].values[0],
                    'ABC_Code':      art_info['ABC_Code'].values[0],
                    'Conso_Futur':   conso_futur
                })
                
    df_merged = pd.DataFrame(records)
    print(f"   ✓ Taille de la matrice finale générée : {len(df_merged)} échantillons d'entraînement.")
    DONNEES_REELLES = True

except Exception as e:
    print(f"⚠️ Erreur lors du chargement des vraies données : {e}")
    print("   Utilisation du mode de secours : génération de données simulées basées sur stats Ch.3")
    DONNEES_REELLES = False

    # Bloc de secours simulé pour éviter le blocage du script
    np.random.seed(42)
    n = 500
    df_merged = pd.DataFrame({
        'Conso_Mois_N':      np.random.poisson(3, n),
        'Conso_Mois_N1':     np.random.poisson(3, n),
        'Conso_Mois_N2':     np.random.poisson(3, n),
        'IPR_AMDEC':         np.random.choice([8,18,36,48], n),
        'Delai_Fournisseur': np.random.choice([14,30,48,90], n),
        'Classe_ABC':        np.random.choice(['A','B','C'], n, p=[0.2,0.3,0.5]),
        'Conso_Futur':       np.random.poisson(4, n)
    })
    df_merged['ABC_Code'] = df_merged['Classe_ABC'].map({'A': 3, 'B': 2, 'C': 1})

# ─── SÉPARATION DES FEATURES ET DE LA CIBLE ───
features_pdr = ['Conso_Mois_N', 'Conso_Mois_N1', 'Conso_Mois_N2', 'IPR_AMDEC', 'Delai_Fournisseur', 'ABC_Code']

X_pdr = df_merged[features_pdr].dropna()
y_pdr = df_merged.loc[X_pdr.index, 'Conso_Futur']

# Split 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(X_pdr, y_pdr, test_size=0.2, random_state=42)

# ─── ENTRAÎNEMENT DE LA RÉGRESSION LINÉAIRE ───
print("\n[3/4] Entraînement du modèle Scikit-Learn...")
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Évaluation des performances
y_pred = model_lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ─── AFFICHAGE DES RÉSULTATS POUR LE RAPPORT DE PFE ───
print("\n" + "=" * 45)
print("✅ MODÈLE ENTRAÎNÉ AVEC SUCCÈS")
print("=" * 45)
print("Coefficients appris (β) :")
for feat, coef in zip(features_pdr, model_lr.coef_):
    print(f"    {feat:25s} : {coef:+.4f}")
print(f"    Intercept (β₀)           : {model_lr.intercept_:+.4f}")

print("\nMétriques d'évaluation sur le jeu de Test :")
print(f"    Erreur Absolue Moyenne (MAE) : {mae:.4f} pièces")
print(f"    Coefficient de dét. (R²)     : {r2:.4f}")

# ─── SAUVEGARDE DE L'ARTEFACT DE PREDICTION ───
print("\n[4/4] Sérialisation du modèle...")
os.makedirs('models', exist_ok=True)
joblib.dump(model_lr, 'models/regression_pdr_model.pkl')
print("✅ Fichier binaire sauvegardé sous : models/regression_pdr_model.pkl\n")