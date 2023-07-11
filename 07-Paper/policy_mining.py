import numpy as np
import pandas as pd

class PolicyMining:
    def __init__(self, file_name, user_attrs, resource_attrs) -> None:
        df_data = pd.read_csv(file_name)
        print("Data loaded!")

        ### number of entities
        n_users = len(df_data[user_attrs].drop_duplicates())
        n_rsrcs = len(df_data[resource_attrs].drop_duplicates())
        print("|U| =", n_users, "\t|R| =", n_rsrcs)

        ### Create an ID for every entity (users and resources)
        user_dict = {}
        for u_idx, u_attr in enumerate(df_data[user_attrs].drop_duplicates().values):
            idx = "ID-" + str(u_idx)
            user_dict[idx] = list(u_attr)
        print("User and resource ID")