# 🏭 Module d'Intelligence Artificielle pour la Maintenance Prédictive

Bienvenue sur le dépôt officiel du code source de mon Projet de Fin d'Études (PFE) d'Ingénieur d'État, réalisé au sein de **Smurfit Westrock Maroc**.

Ce projet s'inscrit dans une démarche d'optimisation de la gestion des pièces de rechange (PDR) sous la méthodologie DMAIC, couplée à la mise en œuvre d'une architecture SCADA et d'algorithmes de Machine Learning.

## 🎯 Objectif du projet
Ce dépôt contient le "cerveau" logiciel du projet. Il a pour but d'analyser en temps réel les dérives thermiques et électriques des variateurs de fréquence (remontées par le système SCADA TIA Portal), et de croiser ces anomalies avec l'historique de consommation pour générer des prédictions d'achat automatisées avant la rupture de stock.

## 🧠 Modèles de Machine Learning utilisés

1. **Isolation Forest (Détection d'anomalies) :** - Entraîné sur les données de télémétrie des équipements.
   - Isole les signatures de pré-défaillance (dérive température/courant).
   
2. **Régression Linéaire Multiple (Prédiction de consommation) :**
   - Entraînée sur l'historique des sorties de magasin, l'IPR (Indice de Priorité du Risque de l'AMDEC) et les délais fournisseurs.
   - Calcule la quantité exacte de pièces de rechange nécessaires.

3. **Module de décision (ROP Dynamique) :**
   - Fusionne les résultats des deux modèles pour ajuster dynamiquement le Point de Commande (ROP) et générer des alertes préventives.

## 📂 Structure du code
- `decision_module.py` : Script principal exécutant les scénarios de test et générant les alertes.
- `regression_pdr.py` : Script d'entraînement et d'évaluation du modèle de prédiction des pièces.
- `metrics_validation_IF.py` : Script d'évaluation des performances du modèle Isolation Forest.
- `models/` : Dossier contenant les modèles sérialisés (`.pkl`) prêts à l'emploi.
- `outputs/graphs/` : Dossier contenant les graphiques générés (courbes d'évolution, matrices de confusion, distribution des résidus).

## 🛠️ Technologies
- **Langage :** Python 3.x
- **Machine Learning :** Scikit-Learn
- **Data Science :** Pandas, NumPy
- **Data Visualization :** Matplotlib, Seaborn

## ⚠️ Clause de Confidentialité
Dans le respect du secret industriel et des politiques de sécurité de **Smurfit Westrock**, les bases de données brutes d'historique de consommation, d'inventaire S004 et de valeurs financières (fichiers `.xlsx` et `.csv`) **ne sont pas incluses dans ce dépôt public**. 
Les scripts sont conçus pour lire ces données en local. Les résultats présentés dans le rapport de PFE attestent du bon fonctionnement des modèles sur les données réelles de l'usine.

---
*Développé par Youness Ouhaddou - Élève Ingénieur en Génie Mécatronique et Systèmes Intelligents à l'ENSA Berrechid.*
*Année Universitaire : 2025/2026.*
