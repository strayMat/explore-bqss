# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from explore_bqss.data import fetch_bqss_data

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)

# %%
# Load data
print("Loading BQSS data...")
data_dict = fetch_bqss_data()
valeur_df = data_dict["valeur"]
finess_df = data_dict["finess"]
metadata_df = data_dict["metadata"]

print(f"Data loaded successfully!")
print(f"Valeur dataset: {valeur_df.shape[0]:,} records")
print(f"FINESS dataset: {finess_df.shape[0]:,} establishments")
print(f"Metadata: {metadata_df.shape[0]} indicators")


# %%
def get_finess_info(finess_id):
    """Get basic information about a FINESS establishment."""
    if hasattr(finess_df, "filter"):  # Polars DataFrame
        finess_info = finess_df.filter(finess_df["num_finess_et"] == finess_id)
        if finess_info.height > 0:
            info = finess_info.head(1).to_pandas().iloc[0]
        else:
            return None
    else:  # Pandas DataFrame
        finess_info = finess_df[finess_df["num_finess_et"] == finess_id]
        if len(finess_info) > 0:
            info = finess_info.iloc[0]
        else:
            return None

    return {
        "finess_id": finess_id,
        "nom": info.get("raison_sociale_et", "N/A"),
        "commune": info.get("commune", "N/A"),
        "departement": info.get("departement", "N/A"),
        "categorie": info.get("libelle_categorie_et", "N/A"),
        "type_etablissement": info.get("type_etablissement", "N/A"),
    }


def get_finess_indicators(finess_id):
    """Get all indicators for a specific FINESS establishment."""
    finess_data = valeur_df[valeur_df["finess"] == finess_id].copy()

    if len(finess_data) == 0:
        return None

    # Merge with metadata to get indicator descriptions
    finess_data = finess_data.merge(
        metadata_df[["name", "title", "description", "type", "source"]], left_on="key", right_on="name", how="left"
    )

    return finess_data


def create_finess_report(finess_id):
    """Create a comprehensive report for a FINESS establishment."""

    print(f"=" * 80)
    print(f"📊 RAPPORT DÉTAILLÉ POUR L'ÉTABLISSEMENT FINESS: {finess_id}")
    print(f"=" * 80)

    # Get establishment info
    finess_info = get_finess_info(finess_id)
    if finess_info is None:
        print(f"❌ Établissement FINESS {finess_id} non trouvé dans la base FINESS")
        return

    print(f"\n🏥 INFORMATIONS GÉNÉRALES")
    print("-" * 40)
    print(f"Nom: {finess_info['nom']}")
    print(f"Commune: {finess_info['commune']}")
    print(f"Département: {finess_info['departement']}")
    print(f"Catégorie: {finess_info['categorie']}")
    print(f"Type: {finess_info['type_etablissement']}")

    # Get indicators data
    finess_data = get_finess_indicators(finess_id)
    if finess_data is None or len(finess_data) == 0:
        print(f"\n❌ Aucune donnée d'indicateurs trouvée pour {finess_id}")
        return

    print(f"\n📈 DONNÉES DISPONIBLES")
    print("-" * 40)
    print(f"Nombre total d'enregistrements: {len(finess_data):,}")
    print(f"Années couvertes: {sorted(finess_data['annee'].unique())}")
    print(f"Nombre d'indicateurs uniques: {finess_data['key'].nunique()}")
    print(f"Types FINESS: {list(finess_data['finess_type'].unique())}")

    # Years analysis
    print(f"\n📅 RÉPARTITION PAR ANNÉE")
    print("-" * 40)
    year_counts = finess_data["annee"].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} indicateurs")

    # Indicators by source
    print(f"\n🔍 INDICATEURS PAR SOURCE")
    print("-" * 40)
    source_counts = finess_data["source"].value_counts()
    for source, count in source_counts.items():
        if pd.notna(source):
            print(f"  {source}: {count:,} indicateurs")

    # Data types analysis
    print(f"\n📊 TYPES DE DONNÉES")
    print("-" * 40)

    # Count non-null values by type
    boolean_count = finess_data["value_boolean"].notna().sum()
    string_count = finess_data["value_string"].notna().sum()
    integer_count = finess_data["value_integer"].notna().sum()
    float_count = finess_data["value_float"].notna().sum()
    date_count = finess_data["value_date"].notna().sum()

    print(f"  Valeurs booléennes: {boolean_count:,}")
    print(f"  Valeurs textuelles: {string_count:,}")
    print(f"  Valeurs entières: {integer_count:,}")
    print(f"  Valeurs décimales: {float_count:,}")
    print(f"  Valeurs dates: {date_count:,}")

    # Top indicators
    print(f"\n🏆 TOP 10 DES INDICATEURS LES PLUS FRÉQUENTS")
    print("-" * 40)
    top_indicators = finess_data["key"].value_counts().head(10)
    for i, (indicator, count) in enumerate(top_indicators.items(), 1):
        # Get indicator title
        title = (
            metadata_df[metadata_df["name"] == indicator]["title"].iloc[0]
            if indicator in metadata_df["name"].values
            else indicator
        )
        print(f"  {i:2d}. {title[:60]}{'...' if len(title) > 60 else ''}")
        print(f"      ({count} enregistrements)")

    # Recent indicators (last year available)
    latest_year = finess_data["annee"].max()
    recent_data = finess_data[finess_data["annee"] == latest_year]

    print(f"\n📋 INDICATEURS RÉCENTS ({latest_year})")
    print("-" * 40)
    print(f"Nombre d'indicateurs pour {latest_year}: {len(recent_data):,}")

    if len(recent_data) > 0:
        # Show some sample recent indicators with values
        sample_recent = recent_data.head(10)
        for _, row in sample_recent.iterrows():
            title = row["title"] if pd.notna(row["title"]) else row["key"]

            # Get the actual value
            value = None
            if pd.notna(row["value_boolean"]):
                value = row["value_boolean"]
            elif pd.notna(row["value_string"]):
                value = row["value_string"]
            elif pd.notna(row["value_integer"]):
                value = row["value_integer"]
            elif pd.notna(row["value_float"]):
                value = row["value_float"]
            elif pd.notna(row["value_date"]):
                value = row["value_date"]
            else:
                value = "Valeur manquante"

            print(f"  • {title[:50]}{'...' if len(title) > 50 else ''}: {value}")

    return finess_data


# %%
# Get a sample FINESS ID to demonstrate
print("🔍 Recherche d'un établissement exemple...")
sample_finess = valeur_df["finess"].value_counts().head(1).index[0]
print(f"Utilisation de l'établissement exemple: {sample_finess}")

# %%
# Create report for sample establishment
sample_report = create_finess_report(sample_finess)


# %%
def create_visualizations(finess_id, finess_data):
    """Create visualizations for the FINESS establishment."""

    if finess_data is None or len(finess_data) == 0:
        print("Aucune donnée à visualiser")
        return

    print(f"\n📊 VISUALISATIONS POUR {finess_id}")
    print("=" * 50)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Analyse des indicateurs pour FINESS {finess_id}", fontsize=16, fontweight="bold")

    # 1. Evolution over years
    year_counts = finess_data["annee"].value_counts().sort_index()
    axes[0, 0].bar(year_counts.index, year_counts.values, color="skyblue", alpha=0.7)
    axes[0, 0].set_title("Nombre d'indicateurs par année")
    axes[0, 0].set_xlabel("Année")
    axes[0, 0].set_ylabel("Nombre d'indicateurs")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Data types distribution
    data_types = {
        "Booléen": finess_data["value_boolean"].notna().sum(),
        "Texte": finess_data["value_string"].notna().sum(),
        "Entier": finess_data["value_integer"].notna().sum(),
        "Décimal": finess_data["value_float"].notna().sum(),
        "Date": finess_data["value_date"].notna().sum(),
    }

    # Remove zero values for pie chart
    data_types = {k: v for k, v in data_types.items() if v > 0}

    if data_types:
        axes[0, 1].pie(data_types.values(), labels=data_types.keys(), autopct="%1.1f%%", startangle=90)
        axes[0, 1].set_title("Répartition des types de données")

    # 3. Sources distribution
    source_counts = finess_data["source"].value_counts()
    if len(source_counts) > 0:
        axes[1, 0].barh(range(len(source_counts)), source_counts.values, color="lightcoral", alpha=0.7)
        axes[1, 0].set_yticks(range(len(source_counts)))
        axes[1, 0].set_yticklabels([str(s)[:20] + "..." if len(str(s)) > 20 else str(s) for s in source_counts.index])
        axes[1, 0].set_title("Indicateurs par source")
        axes[1, 0].set_xlabel("Nombre d'indicateurs")

    # 4. FINESS type distribution
    finess_type_counts = finess_data["finess_type"].value_counts()
    if len(finess_type_counts) > 0:
        axes[1, 1].bar(finess_type_counts.index, finess_type_counts.values, color="lightgreen", alpha=0.7)
        axes[1, 1].set_title("Répartition par type FINESS")
        axes[1, 1].set_xlabel("Type FINESS")
        axes[1, 1].set_ylabel("Nombre d'indicateurs")

    plt.tight_layout()
    plt.show()


# %%
# Create visualizations for the sample establishment
if sample_report is not None:
    create_visualizations(sample_finess, sample_report)


# %%
def analyze_indicator_trends(finess_id, finess_data, indicator_key):
    """Analyze trends for a specific indicator over time."""

    if finess_data is None:
        return

    indicator_data = finess_data[finess_data["key"] == indicator_key]

    if len(indicator_data) == 0:
        print(f"Aucune donnée trouvée pour l'indicateur: {indicator_key}")
        return

    # Get indicator info
    indicator_info = metadata_df[metadata_df["name"] == indicator_key]
    indicator_title = indicator_info["title"].iloc[0] if len(indicator_info) > 0 else indicator_key

    print(f"\n📈 ÉVOLUTION DE L'INDICATEUR: {indicator_title}")
    print("-" * 60)

    # Sort by year
    indicator_data = indicator_data.sort_values("annee")

    # Display evolution
    for _, row in indicator_data.iterrows():
        year = row["annee"]

        # Get the value
        value = None
        if pd.notna(row["value_boolean"]):
            value = row["value_boolean"]
        elif pd.notna(row["value_string"]):
            value = row["value_string"]
        elif pd.notna(row["value_integer"]):
            value = row["value_integer"]
        elif pd.notna(row["value_float"]):
            value = f"{row['value_float']:.2f}"
        elif pd.notna(row["value_date"]):
            value = row["value_date"]
        else:
            value = "N/A"

        print(f"  {year}: {value}")


# %%
# Analyze trends for a specific indicator (if available)
if sample_report is not None and len(sample_report) > 0:
    # Get the most frequent indicator for this establishment
    top_indicator = sample_report["key"].value_counts().index[0]
    analyze_indicator_trends(sample_finess, sample_report, top_indicator)


# %%
# Function to create reports for multiple establishments
def create_comparison_report(finess_list):
    """Create a comparison report for multiple FINESS establishments."""

    print(f"\n{'=' * 80}")
    print(f"📊 RAPPORT COMPARATIF POUR {len(finess_list)} ÉTABLISSEMENTS")
    print(f"{'=' * 80}")

    comparison_data = []

    for finess_id in finess_list:
        finess_info = get_finess_info(finess_id)
        finess_data = get_finess_indicators(finess_id)

        if finess_info and finess_data is not None:
            comparison_data.append({
                "finess_id": finess_id,
                "nom": finess_info["nom"],
                "commune": finess_info["commune"],
                "categorie": finess_info["categorie"],
                "nb_indicateurs": len(finess_data),
                "annees_min": finess_data["annee"].min(),
                "annees_max": finess_data["annee"].max(),
                "nb_sources": finess_data["source"].nunique(),
            })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        print("\n📋 TABLEAU COMPARATIF")
        print("-" * 80)
        print(comparison_df.to_string(index=False))

        # Summary statistics
        print(f"\n📊 STATISTIQUES COMPARATIVES")
        print("-" * 40)
        print(f"Nombre moyen d'indicateurs: {comparison_df['nb_indicateurs'].mean():.1f}")
        print(
            f"Établissement avec le plus d'indicateurs: {comparison_df.loc[comparison_df['nb_indicateurs'].idxmax(), 'nom']}"
        )
        print(f"  → {comparison_df['nb_indicateurs'].max():,} indicateurs")


# %%
# Get top 5 establishments by number of indicators for comparison
print("🔍 Création d'un rapport comparatif...")
top_establishments = valeur_df["finess"].value_counts().head(5).index.tolist()
create_comparison_report(top_establishments)

# %%
print(f"\n{'=' * 80}")
print("✅ ANALYSE TERMINÉE")
print(f"{'=' * 80}")
print("Ce notebook fournit une analyse complète des indicateurs BQSS pour un ou plusieurs établissements FINESS.")
print(
    "Vous pouvez modifier le code pour analyser des établissements spécifiques en changeant la variable 'sample_finess'."
)
