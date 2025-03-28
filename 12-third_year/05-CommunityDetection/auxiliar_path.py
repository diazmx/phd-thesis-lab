### Auxiliar functions

def calculate_log_binning(degree_distribution, n_bins):
    """Compute the log-binning y-values in the degree distribution.

    Divides the degree distribution in `n_bins` segments.

    Parameters
    ----------
    degree_distribution: list
        Network degree distribution.
    n_bins:
        Number of bins to assign.

    Returns
    -------
    (list, list)
        The (x_values, y_values_log_bin_list) tuple.
    """
    current_sum = 0
    previous_k = 0
    y_values_log_bin_list = []
    x_values = []

    for i in range(1, n_bins):
        x_values.append(previous_k)
        current_k = 2 ** (i)
        current_sum = current_sum + current_k
        temp_y_value = sum(degree_distribution[previous_k:current_k])
        temp_y_value = temp_y_value / (current_k-previous_k)
        y_values_log_bin_list.append(temp_y_value)
        previous_k = current_k

        if current_sum > len(degree_distribution):
            x_values.append(previous_k)
            temp_y_value = sum(degree_distribution[previous_k:len(degree_distribution)])
            temp_y_value = temp_y_value / (len(degree_distribution)-previous_k)
            y_values_log_bin_list.append(temp_y_value)            
            break

    return x_values, y_values_log_bin_list

def get_path_topbot(tpe):
    """Return the paht of a type of nodes."""
    
    if tpe:
        return "01-Top"
    else:
        return "02-Bot"

def get_path_dataset(ds):
    """Return the path of a dataset ds."""

    if ds == "AMZ":
        return "01-AMZ"
    elif ds == "HC":
        return "02-HC"
    elif ds == "PM":
        return "03-PM"
    elif ds == "UN":
        return "04-UN"
    elif ds == "TOY":
        return "12-TOY"
    else:
        return None