# J'ai utilisé le dataset Telco Customer Churn de Kaggle, un dataset réel de télécommunications avec 7000 clients. J'ai fait un preprocessing : nettoyage des noms de colonnes, encodage de la variable cible en binaire, et gestion des valeurs manquantes dans TotalCharges.

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "blastchar/telco-customer-churn",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df.columns = df.columns.str.lower().str.replace(" ", "_")
df["churn"] = (df["churn"] == "Yes").astype(int)
df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)

df.to_csv("data/churn_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df['churn'].mean():.1%} churn rate")
print(df.head())

# La dataset Telco — ce qu'elle contient :

# ~7000 clients d'une entreprise télécom
Features : contrat, internet, ancienneté, charges mensuelles...
Target : Churn (parti ou pas)#
