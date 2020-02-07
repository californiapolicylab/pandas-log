import itertools
from inspect import signature

import humanize
import pandas as pd


def _get_pretty_param(param_with_default, arg_value):
    res = str(param_with_default)
    if arg_value is not None:
        param_name = res.split("=")[0]
        arg_value = (
            f'"{arg_value}"' if isinstance(arg_value, str) else arg_value
        )
        res = (
            param_name
            if isinstance(arg_value, pd.DataFrame) or isinstance(arg_value, pd.Series)
            else f"{param_name}={arg_value}"
        )
    return res


def get_params(method, full_signature=True):
    return [
        param_value if full_signature else param_name
        for param_name, param_value in signature(
            method
        ).parameters.items()
        if param_name not in ("kwargs", "self")
    ]


def bold(text):
    return f"\033[1m{text}\033[0m"


def get_pretty_signature_repr(method, args, full_signature=True):
    """ Get the signature for the original pandas method with actual values

        :param cls: the pandas class
        :param fn: The pandas method
        :param args: The arguments used when it was applied
        :return: string representation of the signature for the applied pandas method
    """
    zip_func = itertools.zip_longest if full_signature else zip
    params = get_params(method, full_signature=full_signature)
    args_vals = ", ".join(
        _get_pretty_param(param_with_default, arg_value)
        for param_with_default, arg_value in zip_func(params, args)
    )
    return f"{bold(method.__name__)}({args_vals}):"


def calc_df_series_memory(pd_obj):
    mem = pd_obj.memory_usage(index=True, deep=True)
    if isinstance(pd_obj, pd.DataFrame):
        mem = mem.sum()
    return humanize.naturalsize(mem)
