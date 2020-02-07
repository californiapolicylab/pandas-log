# -*- coding: utf-8 -*-

"""Main module."""

import warnings
from contextlib import contextmanager
from functools import wraps
from copy import copy
import humanize

import pandas as pd
import pandas_flavor as pf

from pandas_log import settings
from pandas_log.aop_utils import (keep_pandas_func_copy,
                                  restore_pandas_func_copy, get_pretty_signature_repr)
from pandas_log.pandas_execution_stats import StepStats, get_execution_stats
from pandas_log import patched_log_functions
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


def auto_disable():
    _disable_pandas_log()


@contextmanager
def disable():
    _disable_pandas_log()
    yield
    _enable_pandas_log()


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


def wrap_logging(original_method, logging_function):
    @wraps(original_method)
    def wrapper(*args, **fn_kwargs):
        input_pd_obj, fn_args = args[0], args[1:]
        global PANDAS_LOG_PARAMETERS
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


def patch_pandas_method(cls, method_name, logging_function, patched_prefix='patched_', original_prefix='original_'):

    original_method = getattr(cls, method_name)
    patched_method = wrap_logging(original_method, logging_function)
    setattr(cls, original_prefix + method_name, original_method)
    setattr(cls, patched_prefix + method_name, patched_method)

    def delegation_method(self, *args, **kwargs):
        global PANDAS_LOG_ON
        if PANDAS_LOG_ON:
            return getattr(self, patched_prefix + method_name)(*args, **kwargs)
        else:
            return getattr(self, original_prefix + method_name)(*args, **kwargs)

    setattr(cls, method_name, delegation_method)


def initialize_pandas_log():
    for method_name in settings.DATAFRAME_METHODS_TO_OVERIDE + settings.DATAFRAME_VERBOSE_METHODS_TO_OVERIDE:
        logging_function = getattr(patched_log_functions, 'log_' + method_name, 'log_default')
        patch_pandas_method(pd.DataFrame, method_name, logging_function)
    global PANDAS_LOG_INITIALIZED
    PANDAS_LOG_INITIALIZED = True


def calc_df_series_memory(pd_obj):
    mem = pd_obj.memory_usage(index=True, deep=True)
    if isinstance(pd_obj, pd.DataFrame):
        mem = mem.sum()
    return humanize.naturalsize(mem)


def execute_method_with_stats(method, input_pd_obj, fn_args, fn_kwargs, parameters):
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


def enable_extras():
    import pandas_log.extras


def create_overide_pandas_func(cls, func, verbose, silent, full_signature, copy_ok, calculate_memory):
    """ Create overridden pandas method dynamically with
        additional logging using DataFrameLogger

        Note: if we extracting _overide_pandas_method outside we need to implement decorator like here
              https://stackoverflow.com/questions/10176226/how-do-i-pass-extra-arguments-to-a-python-decorator

        :param cls: pandas class for which the method should be overriden
        :param func: pandas method name to be overridden
        :param silent: Whether additional the statistics get printed
        :param full_signature: adding additional information to function signature
        :param copy_ok: whether the dataframe is allowed to be copied to calculate more informative metadata logs_and_tips
        :return: the same function with additional logging capabilities
    """

    def _run_method_and_calc_stats(
        fn, fn_args, fn_kwargs, input_df, full_signature, silent, verbose, copy_ok, calculate_memory
    ):

        if copy_ok:
            # If we're ok to make copies, copy the input_pd_obj so that we can compare against the output of inplace methods
            try:
                # Will hit infinite recursion if we use the patched copy method so use the original
                original_input_df = getattr(input_df, settings.ORIGINAL_METHOD_PREFIX+'copy')(deep=True)
            except AttributeError:
                original_input_df = input_df.copy(deep=True)
        output_df, execution_stats = get_execution_stats(
            cls, fn, input_df, fn_args, fn_kwargs, calculate_memory
        )
        if output_df is None:
            # The operation was strictly in place so we just call the dataframe the output_df as well
            output_df = input_df
        if copy_ok:
            # If this isn't true and the method was strictly inplace, input_pd_obj and output_df will just
            # point to the same object
            input_df = original_input_df

        step_stats = StepStats(
            execution_stats,
            cls,
            fn,
            fn_args,
            fn_kwargs,
            full_signature,
            input_df,
            output_df,
        )
        step_stats.log_stats_if_needed(silent, verbose, copy_ok)
        if isinstance(output_df, pd.DataFrame) or isinstance(
            output_df, pd.Series
        ):
            step_stats.persist_execution_stats()
        # Don't think there's any garbage collection we should do manually?
        return output_df

    def _overide_pandas_method(fn):
        if cls == pd.DataFrame:
            register_method_wrapper = pf.register_dataframe_method
        elif cls == pd.Series:
            register_method_wrapper = pf.register_series_method
        @register_method_wrapper
        @wraps(fn)
        def wrapped(*args, **fn_kwargs):

            input_df, fn_args = args[0], args[1:]
            output_df = _run_method_and_calc_stats(
                fn,
                fn_args,
                fn_kwargs,
                input_df,
                full_signature,
                silent,
                verbose,
                copy_ok,
                calculate_memory
            )
            return output_df

        return wrapped

    return exec(
        f"@_overide_pandas_method\ndef {func}(df, *args, **kwargs): pass"
    )


if __name__ == "__main__":
    pass
