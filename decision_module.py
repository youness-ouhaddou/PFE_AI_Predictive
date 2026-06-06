import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore') # Désactive les petits avertissements Scikit-learn

os.makedirs('outputs/graphs', exist_ok=True)

# ─── Chargement des modèles ───
try:
    model_if = joblib.load('models/isolation_forest_model.pkl')
    scaler   = joblib.load('models/scaler_if.pkl')
    model_lr = joblib.load('models/regression_pdr_model.pkl')
except Exception as e:
    print(f"❌ Erreur de chargement des modèles : {e}")
    exit()

def analyse_complete(temp, courant, tension, run_hours, fault_count,
                     stock_disponible, piece_designation,
                     rop_calcule, delai_fournisseur,
                     historique_conso,
                     ipr_amdec, classe_abc):
    
    # ═══ ÉTAPE 1 : Isolation Forest ═══
    donnee       = scaler.transform([[temp, courant, tension, run_hours, fault_count]])
    prediction   = model_if.predict(donnee)[0]
    score        = model_if.score_samples(donnee)[0]
    anomalie     = (prediction == -1)

    print(f"\n📡 DONNÉES VARIATEUR (source SCADA) :")
    print(f"   Température   : {temp:.1f}°C")
    print(f"   Courant       : {courant:.2f}A")
    print(f"   Tension       : {tension:.1f}V")
    print(f"   Heures marche : {run_hours:.0f}h")
    print(f"   Nb défauts    : {fault_count}")

    print(f"\n🤖 MODÈLE 1 — Isolation Forest :")
    print(f"   Score IF       : {score:.4f}")
    print(f"   Résultat       : {'⚠️ ANOMALIE DÉTECTÉE' if anomalie else '✅ NORMAL'}")

    if not anomalie:
        print("\n✅ Variateur normal — Aucune action requise sur le stock.")
        return "NORMAL", 0, score

    # ═══ ÉTAPE 2 : Régression Linéaire ═══
    abc_code      = {'A': 3, 'B': 2, 'C': 1}.get(classe_abc, 1)
    
    # On prend toujours les 3 dernières valeurs de l'historique
    hist_recent = historique_conso[-3:]
    features_in   = np.array([[
        hist_recent[2], hist_recent[1], hist_recent[0], # N, N-1, N-2
        ipr_amdec, delai_fournisseur, abc_code
    ]])
    conso_predite = max(0, round(model_lr.predict(features_in)[0]))

    print(f"\n📊 MODÈLE 2 — Régression Linéaire (PDR) :")
    print(f"   Consommation prédite : {conso_predite} pièces")

    # ═══ ÉTAPE 3 : Décision combinée (ROP Dynamique) ═══
    stock_necessaire = rop_calcule + conso_predite

    print(f"\n📦 ANALYSE STOCK PDR (source Excel S004) :")
    print(f"   Pièce critique        : {piece_designation}")
    print(f"   Stock actuel          : {stock_disponible} unités")
    print(f"   ROP statique (Ch.3)   : {rop_calcule} unités")
    print(f"   Prédiction IA         : +{conso_predite} unités")
    print(f"   ─────────────────────────────────────────")
    print(f"   Stock nécessaire      : {stock_necessaire} unités")

    # ═══ ÉTAPE 4 : Niveau d'alerte ═══
    if stock_disponible == 0:
        niveau = "CRITIQUE"
        print(f"\n🚨 ALERTE CRITIQUE : Stock = 0 | Panne imminente possible")
    elif stock_disponible < stock_necessaire:
        niveau  = "ATTENTION"
        print(f"\n⚠️ ALERTE PRÉVENTIVE : Stock ({stock_disponible}) < Besoin ({stock_necessaire})")
    else:
        niveau = "WATCH"
        print(f"\n👁️ SURVEILLANCE : Stock ({stock_disponible}) ≥ Besoin ({stock_necessaire})")
        
    return niveau, conso_predite, score

if __name__ == "__main__":
    # =====================================================================
    # 6.5.1 EXÉCUTION DES TROIS TESTS
    # =====================================================================
    
    print("\n" + "█"*60)
    print("TEST 1 : SCÉNARIO FONCTIONNEMENT NORMAL")
    print("█"*60)
    niveau_1, pred_1, score_1 = analyse_complete(
        temp=47.2, courant=11.8, tension=400.0, run_hours=2500, fault_count=0,
        stock_disponible=5, piece_designation="Carte CU320 SINAMICS",
        rop_calcule=2, delai_fournisseur=48, historique_conso=[1, 1, 2, 1, 2, 1], ipr_amdec=48, classe_abc='A'
    )

    print("\n" + "█"*60)
    print("TEST 2 : ANOMALIE CRITIQUE + RUPTURE STOCK")
    print("█"*60)
    niveau_2, pred_2, score_2 = analyse_complete(
        temp=72.5, courant=17.8, tension=392.0, run_hours=8500, fault_count=4,
        stock_disponible=0, piece_designation="Carte CU320 SINAMICS",
        rop_calcule=2, delai_fournisseur=48, historique_conso=[1, 2, 1, 3, 2, 1], ipr_amdec=48, classe_abc='A'
    )

    print("\n" + "█"*60)
    print("TEST 3 : DÉGRADATION + STOCK AU ROP")
    print("█"*60)
    niveau_3, pred_3, score_3 = analyse_complete(
        temp=61.0, courant=15.5, tension=397.0, run_hours=6200, fault_count=2,
        stock_disponible=2, piece_designation="Carte CU320 SINAMICS",
        rop_calcule=2, delai_fournisseur=48, historique_conso=[1, 1, 2, 2, 3, 2], ipr_amdec=48, classe_abc='A'
    )

    print("\n" + "═"*60)
    print("   RÉSUMÉ DES RÉSULTATS POUR LE TABLEAU 6.5.2")
    print("═"*60)
    print(f"Normal      | Score IF: {score_1:.2f} | IA: {pred_1} pc | Alerte: {niveau_1}")
    print(f"Critique    | Score IF: {score_2:.2f} | IA: {pred_2} pc | Alerte: {niveau_2}")
    print(f"Dégradation | Score IF: {score_3:.2f} | IA: {pred_3} pc | Alerte: {niveau_3}")


    # =====================================================================
    # 6.5.3 GRAPHIQUE ÉVOLUTION TEMPORELLE
    # =====================================================================
    print("\n⏳ Génération du graphique d'évolution...")
    t = np.linspace(0, 80, 500)
    temp_sim    = np.where(t < 60, 45.0 + t * 0.55, 80.0)
    current_sim = np.where(t < 60, 11.5 + t * 0.10, 0.0)

    def score_thermique(temp):
        if temp < 50: return 0
        elif temp < 60: return 20
        elif temp < 68: return 55
        elif temp < 75: return 80
        else: return 100

    scores_viz = [
        score_thermique(T) * 0.40 + min(max((C - 11.5) / 6.5 * 100, 0), 100) * 0.25
        for T, C in zip(temp_sim, current_sim)
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        'Évolution temporelle — Scénario dégradation progressive → panne\n'
        'Source : FC_Simulator (TIA Portal V17) — Scénarios 1 et 2',
        fontsize=12, fontweight='bold'
    )

    ax1.plot(t, temp_sim, color='#e74c3c', linewidth=2)
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.8, label='Seuil Watch 60°C')
    ax1.axhline(y=75, color='red', linestyle='--', alpha=0.8, label='Seuil Critique 75°C')
    ax1.axvspan(60, 80, color='red', alpha=0.1, label='Zone panne')
    ax1.set_ylabel('Température (°C)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, current_sim, color='#3498db', linewidth=2)
    ax2.axhline(y=15, color='orange', linestyle='--', alpha=0.8, label='Seuil Watch 15A')
    ax2.axhline(y=18, color='red', linestyle='--', alpha=0.8, label='Seuil Critique 18A')
    ax2.axvspan(60, 80, color='red', alpha=0.1)
    ax2.set_ylabel('Courant (A)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, scores_viz, color='#9b59b6', linewidth=2.5)
    ax3.axhline(y=35, color='orange', linestyle='--', linewidth=1.5, label='Seuil Level 1 — Watch')
    ax3.axhline(y=65, color='red', linestyle='--', linewidth=1.5, label='Seuil Level 2 — Critique')
    ax3.fill_between(t, scores_viz, 35, where=[s > 35 for s in scores_viz], color='orange', alpha=0.25, label='Zone Watch')
    ax3.fill_between(t, scores_viz, 65, where=[s > 65 for s in scores_viz], color='red', alpha=0.25, label='Zone Critique')
    ax3.set_ylabel('Score Risque Global', fontsize=10)
    ax3.set_xlabel('Temps (secondes)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/graphs/evolution_degradation.png', dpi=150, bbox_inches='tight')
    print("✅ Graphique 'evolution_degradation.png' sauvegardé dans le dossier outputs/graphs")
    plt.show()