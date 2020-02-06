# -*- coding: utf-8 -*-

"""Main module."""

import warnings
from contextlib import contextmanager
from functools import wraps
from copy import copy

import pandas as pd
import pandas_flavor as pf

from pandas_log import settings
from pandas_log.aop_utils import (keep_pandas_func_copy,
                                  restore_pandas_func_copy,)
from pandas_log.pandas_execution_stats import StepStats, get_execution_stats
import pandas_log.patched_logs_functions as plf

__all__ = ["auto_enable", "auto_disable", "enable"]


ALREADY_ENABLED = False
global PANDAS_LOG_ON
PANDAS_LOG_ON = True



def disable():
    global PANDAS_LOG_ON
    PANDAS_LOG_ON = False


def auto_disable():
    """ Restore original pandas method without the additional log functionality (statistics)
        Note: we keep the original methods using original_ prefix.
        :return: None
    """
    global ALREADY_ENABLED
    if not ALREADY_ENABLED:
        return

    for cls in (pd.DataFrame, pd.Series):
        for func in dir(cls):
            if func.startswith(settings.ORIGINAL_METHOD_PREFIX):
                restore_pandas_func_copy(cls, func)
    ALREADY_ENABLED = False


@contextmanager
def enable(verbose=False, silent=False, full_signature=True, copy_ok=True, calculate_memory=False):
    """ Adds the additional logging functionality (statistics) to pandas methods only for the scope of this
        context manager.

        :param verbose: Whether some inner functions should be recorded as well.
                        For example: when a dataframe being copied
        :param silent: Whether additional the statistics get printed
        :param full_signature: adding additional information to function signature
        :param copy_ok: whether the dataframe is allowed to be copied to calculate more informative metadata logs
        :return: None
    """

    auto_enable(verbose, silent, full_signature, copy_ok, calculate_memory)
    yield
    auto_disable()


def auto_enable(verbose=False, silent=False, full_signature=True, copy_ok=True, extras=True, calculate_memory=False):
    """ Adds the additional logging functionality (statistics) to pandas methods.

        :param verbose: Whether some inner functions should be recorded as well.
                        For example: when a dataframe being copied
        :param silent: Whether additional the statistics get printed
        :param full_signature: adding additional information to function signature
        :param copy_ok: whether the dataframe is allowed to be copied to calculate more informative metadata logs
        :return: None
    """
    global ALREADY_ENABLED
    if ALREADY_ENABLED:
        return

    settings.DATAFRAME_METHODS_TO_OVERIDE.extend(
        settings.DATAFRAME_ADDITIONAL_METHODS_TO_OVERIDE
    )

    # Suppressing warning of the fact we override pandas functions.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for func in dir(pd.DataFrame):
            if func in settings.DATAFRAME_METHODS_TO_OVERIDE:
                keep_pandas_func_copy(pd.DataFrame, func)
                create_overide_pandas_func(
                    pd.DataFrame, func, verbose, silent, full_signature, copy_ok, calculate_memory
                )
        for func in dir(pd.Series):
            if func in settings.SERIES_METHODS_TO_OVERIDE:
                keep_pandas_func_copy(pd.Series, func)
                create_overide_pandas_func(
                    pd.Series, func, verbose, silent, full_signature, copy_ok, calculate_memory
                )
    if extras:
        enable_extras()
    ALREADY_ENABLED = True


def override_dataframe_method(method_name, patched_function, patched_prefix='patched_', original_prefix='original_'):

    original_method = getattr(pd.DataFrame, method_name)
    setattr(pd.DataFrame, patched_prefix + method_name, patched_function)
    setattr(pd.DataFrame, original_prefix + method_name, original_method)

    def delegation_method(self, *args, **kwargs):
        global PANDAS_LOG_ON
        if PANDAS_LOG_ON:
            return getattr(self, patched_prefix + method_name)(*args, **kwargs)
        else:
            return getattr(self, original_prefix + method_name)(*args, **kwargs)

    setattr(pd.DataFrame, method_name, delegation_method)


def init_pandas_log():
    for method_name in settings.DATAFRAME_METHODS_TO_OVERIDE:
        patched_function = getattr(plf, 'log_' + method_name, 'log_default')
        override_dataframe_method(method_name, patched_function)


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
        :param copy_ok: whether the dataframe is allowed to be copied to calculate more informative metadata logs
        :return: the same function with additional logging capabilities
    """

    def _run_method_and_calc_stats(
        fn, fn_args, fn_kwargs, input_df, full_signature, silent, verbose, copy_ok, calculate_memory
    ):

        if copy_ok:
            # If we're ok to make copies, copy the input_df so that we can compare against the output of inplace methods
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
            # If this isn't true and the method was strictly inplace, input_df and output_df will just
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
