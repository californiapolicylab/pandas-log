# -*- coding: utf-8 -*-

"""Main module."""

from contextlib import contextmanager
from functools import wraps
from copy import copy
import humanize

import pandas as pd
import pandas_flavor as pf

from pandas_log import settings
from pandas_log.utils import get_pretty_signature_repr, calc_df_series_memory
from pandas_log import logging_functions
from time import time

__all__ = ["auto_enable", "auto_disable", "enable"]


PANDAS_LOG_INITIALIZED = False
PANDAS_LOG_ON = False
PANDAS_LOG_PARAMETERS = {}


def _disable_pandas_log():
    global PANDAS_LOG_ON
    PANDAS_LOG_ON = False


def _enable_pandas_log():
    global PANDAS_LOG_ON
    PANDAS_LOG_ON = True


def enable_extras():
    pass


def initialize_pandas_log():
    to_override = {
        pd.DataFrame: settings.DATAFRAME_METHODS_TO_OVERIDE + settings.DATAFRAME_VERBOSE_METHODS_TO_OVERIDE,
        pd.Series: settings.SERIES_METHODS_TO_OVERIDE
    }
    for cls, method_names in to_override.items():
        for method_name in method_names:
            logging_function = getattr(logging_functions, 'log_' + method_name, 'log_default')
            patch_pandas_method(cls, method_name, logging_function)
    global PANDAS_LOG_INITIALIZED
    PANDAS_LOG_INITIALIZED = True


@contextmanager
def enable(verbose=False, log_full_signature=True, copy_ok=True, extras=True, calculate_memory=False):
    """ Adds the additional logging functionality (statistics) to pandas methods only for the scope of this
        context manager.

        :param verbose: Whether some inner functions should be recorded as well.
                        For example: when a dataframe being copied
        :param full_signature: adding additional information to function signature
        :param copy_ok: whether the dataframe is allowed to be copied to calculate more informative metadata logs_and_tips
        :return: None
    """
    global PANDAS_LOG_ON
    global PANDAS_LOG_INITIALIZED
    global PANDAS_LOG_PARAMETERS
    PANDAS_LOG_PARAMETERS = {
        'verbose': verbose,
        'log_full_signature': log_full_signature,
        'copy_ok': copy_ok,
        'calculate_memory': calculate_memory
    }
    if extras:
        enable_extras()
    if PANDAS_LOG_ON:
        return
    if not PANDAS_LOG_INITIALIZED:
        initialize_pandas_log()

    _enable_pandas_log()
    yield
    _disable_pandas_log()


def auto_enable(verbose=False, log_full_signature=True, copy_ok=True, extras=True, calculate_memory=False):
    """ Adds the additional logging functionality (statistics) to pandas methods.

        :param verbose: Whether some inner functions should be recorded as well.
                        For example: when a dataframe being copied
        :param silent: Whether additional the statistics get printed
        :param log_full_signature: adding additional information to function signature
        :param copy_ok: whether the dataframe is allowed to be copied to calculate more informative metadata logs_and_tips
        :return: None
    """
    global PANDAS_LOG_ON
    global PANDAS_LOG_INITIALIZED
    global PANDAS_LOG_PARAMETERS
    PANDAS_LOG_PARAMETERS = {
        'verbose': verbose,
        'log_full_signature': log_full_signature,
        'copy_ok': copy_ok,
        'calculate_memory': calculate_memory
    }
    if extras:
        enable_extras()
    if PANDAS_LOG_ON:
        return
    if not PANDAS_LOG_INITIALIZED:
        initialize_pandas_log()
    _enable_pandas_log()


@contextmanager
def disable():
    _disable_pandas_log()
    yield
    _enable_pandas_log()


def auto_disable():
    _disable_pandas_log()


def patch_pandas_method(cls, method_name, logging_function, patched_prefix='patched_', original_prefix='original_'):
    original_method = getattr(cls, method_name)
    patched_method = wrap_with_logging(original_method, logging_function)
    setattr(cls, original_prefix + method_name, original_method)
    setattr(cls, patched_prefix + method_name, patched_method)

    def delegation_method(self, *args, **kwargs):
        global PANDAS_LOG_ON
        if PANDAS_LOG_ON:
            return getattr(self, patched_prefix + method_name)(*args, **kwargs)
        else:
            return getattr(self, original_prefix + method_name)(*args, **kwargs)

    setattr(cls, method_name, delegation_method)


def wrap_with_logging(original_method, logging_function):
    @wraps(original_method)
    def wrapper(*args, **fn_kwargs):
        input_pd_obj, fn_args = args[0], args[1:]
        global PANDAS_LOG_PARAMETERS
        if (original_method.__name__ in settings.DATAFRAME_VERBOSE_METHODS_TO_OVERIDE and
                not PANDAS_LOG_PARAMETERS['verbose'] and
                isinstance(input_pd_obj, pd.DataFrame)):
            # This method doesn't get logged so we just return the output and skip the body
            return original_method(*args, **fn_kwargs)
        output_pd_obj, stats = execute_method_with_stats(
            original_method,
            input_pd_obj,
            fn_args,
            fn_kwargs,
            PANDAS_LOG_PARAMETERS
        )
        logs_and_tips = logging_function(output_pd_obj, input_pd_obj, *fn_args, **fn_kwargs)
        print_logs_and_stats(
            method=original_method,
            args=fn_args,
            stats=stats,
            logs_and_tips=logs_and_tips,
            parameters=PANDAS_LOG_PARAMETERS
        )
        return output_pd_obj
    return wrapper


def execute_method_with_stats(method, input_pd_obj, fn_args, fn_kwargs, parameters):
    if method.__name__ in settings.DATAFRAME_VERBOSE_METHODS_TO_OVERIDE and isinstance(input_pd_obj, pd.DataFrame):
    # Shouldn't log this method so just return empty logs and skip the body of the function
    start = time()
    output_pd_obj = method(input_pd_obj, *fn_args, **fn_kwargs)
    exec_time = time() - start

    exec_time_pretty = humanize.naturaldelta(exec_time)
    if exec_time_pretty == "a moment":
        exec_time_pretty = f"{round(exec_time,6)} seconds"

    input_memory_size = calc_df_series_memory(input_pd_obj) if parameters['calculate_memory'] else None
    output_memory_size = calc_df_series_memory(output_pd_obj) if parameters['calculate_memory'] else None

    execution_stats = {
        'exec_time': exec_time_pretty,
        'input_memory_size': input_memory_size,
        'output_memory_size': output_memory_size
    }
    return output_pd_obj, execution_stats


def print_logs_and_stats(method, args, logs_and_tips, stats, parameters):
    method_signature = get_pretty_signature_repr(
        method, args, parameters['log_full_signature']
    )

    # Step Metadata stats
    logs, tips = logs_and_tips[0], logs_and_tips[1]
    if not logs:
        # This method isn't patched and verbose is false so we don't print the default
        return ''
    else:
        formatted_logs = f"\033[4mMetadata\033[0m:\n{logs}" if logs else ""
        formatted_tips = f"\033[4mTips\033[0m:\n{tips}" if tips else ""

        # Step Execution stats
        exec_time_humanize = (
            f"* Execution time: Step Took {stats['exec_time']}."
        )
        exec_stats_raw = [exec_time_humanize]
        if stats['input_memory_size'] is not None:
            exec_stats_raw.append(f"* Input Dataframe size is {stats['input_memory_size']}.")
        if stats['output_memory_size'] is not None:
            exec_stats_raw.append(f"* Output Dataframe size is {stats['output_memory_size']}.")
        exec_stats_raw_str = '\n\t'.join(exec_stats_raw)
        execution_stats = f"\033[4mExecution Stats\033[0m:\n\t{exec_stats_raw_str}"

        print(f"{method_signature}\n\t{formatted_logs}\n\t{execution_stats}\n\t{formatted_tips}\n")
