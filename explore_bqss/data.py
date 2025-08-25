from pathlib import Path

import pandas as pd
import polars as pl

from explore_bqss.constants import DIR2DATA


def fetch_bqss_data(cache_dir: Path = DIR2DATA) -> dict:
    """
    Fetch the data from the data.gouv.fr website if not already in the cache_dir, otherwise fetch them from the cache_dir.

    Args:
        cache_dir (str, optional): _description_. Defaults to DIR2DATA.

    Returns:
        dict: _description_
    """
    df_metadata = [
        {
            "df_name": "valeur",
            "url": "https://www.data.gouv.fr/fr/datasets/r/da295ccc-2011-4995-b810-ad1f1a82a83f",
        },
        {
            "df_name": "finess",
            "url": "https://www.data.gouv.fr/fr/datasets/r/b8ef14e5-4f98-4596-82ed-44881c65a0cc",
        },
        {
            "df_name": "metadata",
            "url": "https://www.data.gouv.fr/fr/datasets/r/e56fb5a5-5a74-4507-ba77-e0411d4aa234",
        },
    ]

    df_dict = {}
    for df_metadata_dict in df_metadata:
        df_name = df_metadata_dict["df_name"]
        print(f"Fetching {df_name} data")
        if df_name != "finess":
            local_path = cache_dir / f"{df_name}.csv"
            if not local_path.exists():
                df = pd.read_csv(df_metadata_dict["url"], low_memory=False)
                df.to_csv(local_path, index=False)
            else:
                df = pd.read_csv(local_path, low_memory=False)
        else:
            local_path = cache_dir / f"{df_name}.parquet"
            if not local_path.exists():
                df = pd.read_parquet(df_metadata_dict["url"])
                df.to_parquet(local_path)
            else:
                df = pl.read_parquet(local_path)
        df_dict[df_name] = df
    return df_dict
