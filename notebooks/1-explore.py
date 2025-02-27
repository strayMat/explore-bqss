# %%

from explore_bqss.data import fetch_bqss_data

# %%
data_dict = fetch_bqss_data()
valeur_df = data_dict["valeur"]
finess_df = data_dict["finess"]
metadata = data_dict["metadata"]
# %%
