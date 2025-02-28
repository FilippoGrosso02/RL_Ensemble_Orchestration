import copy
from math import log
import json
from rohe.common import rohe_utils
import pandas as pd

DEFAULT_ZERO_ERROR_EVA = float("1e-20")

# This script derives from the ROHE directory


def map_to_log_scale(value, min_value, max_value, logbase):
    if logbase == 1:
        # Linear mapping
        return (value - min_value) / (max_value - min_value)
    
    if value < min_value:
        # Map values lower than min_value to a negative logarithmic scale
        normalized_value = (value - min_value) / (max_value - min_value)
        log_scaled_value = -log(-normalized_value * (logbase - 1) + 1, logbase)
        return log_scaled_value
    elif value > max_value:
        # Map values higher than max_value to a logarithmic scale greater than 1
        normalized_value = (value - min_value) / (max_value - min_value)
        log_scaled_value = log(normalized_value * (logbase - 1) + 1, logbase)
        return log_scaled_value
    else:
        # Normalize the value to the range [min_value, max_value]
        normalized_value = (value - min_value) / (max_value - min_value)
        # Apply the logarithmic scale
        log_scaled_value = log(normalized_value * (logbase - 1) + 1, logbase)
        return log_scaled_value


def map_to_linear_scale(value, min_value, max_value):
    # Map the linear value to the range [0, 1]
    mapped_value = (value - min_value) / (max_value - min_value)

    return mapped_value


def calculate_statistic(column, statistic):
    if statistic == "max":
        return column.max()
    elif statistic == "min":
        return column.min()
    elif statistic == "sum":
        return column.sum()
    elif statistic in ("avg", "mean"):
        return column.mean()
    elif statistic == "prod":
        return column.prod()
    elif statistic == "rprod":
        new_col = 1 - column
        return 1 - new_col.prod()
    else:
        raise ValueError(
            "Invalid statistic. Valid options are 'prod', 'rprod', sum, 'max', 'min', or 'avg'."
        )


def calculate_scaled_value(value, max_value, min_value, objective, scale, logbase=2):
    if scale == "log":
        scaled_value = map_to_log_scale(value, min_value, max_value, logbase)
    elif scale == "linear":
        scaled_value = map_to_linear_scale(value, min_value, max_value)
    else:
        return None
    if objective == "max":
        return scaled_value
    elif objective == "min":
        return 1 - scaled_value
    return None


# Not used for now
def score_estimation(ensemble: list, contract: dict):
    """
    contract example:
    {
        "mlSpecific":{
            "missRateOfClass1and6":{
            "operator": "prod",
            "weight": 1,
            "min_value": 0.00001,
            "max_value": 0.1,
            "objective": "min",
            "scale": "log",
            "logbase": 2
            },
            "missRateOfClass1": {
            "operator": "prod",
            "weight": 1,
            "min_value": 0.00001,
            "max_value": 0.1,
            "objective": "min",
            "scale": "log"
            },
            "generalAccuracy": {
            "operator": "rprod",
            "weight": 1,
            "min_value": 0.6,
            "max_value": 0.99,
            "objective": "max",
            "scale": "log"
            },
            "confidenceOnClass1": {
            "operator": "avg",
            "weight": 1,
            "min_value": 0.5,
            "max_value": 0.95,
            "objective": "max",
            "scale": "log"
            }
        }
    }
    """

    performance_df = pd.DataFrame()
    for ml_service in ensemble:
        row = pd.DataFrame([ml_service.to_dict()])
        performance_df = pd.concat([performance_df, row], ignore_index=True)
    total_score = 0
    metrics = copy.deepcopy(contract["mlSpecific"])
    result = {}
    for metric_key, metric in metrics.items():
        agg_metric = calculate_statistic(performance_df[metric_key], metric["operator"])
        if "logbase" in metric:
            sub_score = calculate_scaled_value(
                agg_metric,
                metric["max_value"],
                metric["min_value"],
                metric["objective"],
                metric["scale"],
                logbase=metric["logbase"],
            )
        else:
            sub_score = calculate_scaled_value(
                agg_metric,
                metric["max_value"],
                metric["min_value"],
                metric["objective"],
                metric["scale"],
            )
        sub_score *= metric["weight"]
        total_score += sub_score
        result[metric_key] = agg_metric
        result[metric_key + "_score"] = sub_score
        # print(metric_key,": ",agg_metric, " ; Sub-score: ",sub_score)
    result["total_score"] = total_score
    return result


def rl_reward_estimation(performance_metrics: dict, contract_path):
    contract = rohe_utils.load_config(contract_path)

    total_score = 0
    
    metrics = copy.deepcopy(contract["mlSpecific"])    
    for metric_key, metric in metrics.items():
        agg_metric = performance_metrics[metric_key]
        if "logbase" in metric:
            sub_score = calculate_scaled_value(
                agg_metric,
                metric["max_value"],
                metric["min_value"],
                metric["objective"],
                metric["scale"],
                logbase=metric["logbase"],
            )
        else:
            sub_score = calculate_scaled_value(
                agg_metric,
                metric["max_value"],
                metric["min_value"],
                metric["objective"],
                metric["scale"],
            )
        sub_score *= metric["weight"]
        total_score += sub_score
        
        # print(metric_key,": ",agg_metric, " ; Sub-score: ",sub_score)
    
    return total_score