def add_new_index(data, attrs, type="U"):
    # Create an ID for every entity (users and resources)
    if type == "U":
        user_ids_vals = [
            idx for idx in list(data[attrs].drop_duplicates().index)]
    else:
        user_ids_vals = [
            idx+10000 for idx in list(data[attrs].drop_duplicates().index)]
    user_ids_keys = [
        str(i) for i in data[attrs].drop_duplicates().values]
    user_ids_dict = dict(zip(user_ids_keys, user_ids_vals))

    # Create the ID column in the DF
    list_usr_idx = []
    for log in data[attrs].values:
        key_idx = str(log)
        list_usr_idx.append(user_ids_dict[key_idx])
    data[type+"ID"] = list_usr_idx
    return data
