import functools
import time

from .. import logger


def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = ["{k}={v!r}".format(k=k, v=v) for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print("Calling {}({})".format(func.__name__, signature))
        value = func(*args, **kwargs)
        print("{} returned {}".format(func.__name__, value))
        return value
    return wrapper_debug


def tracker(_func=None, *, ulogger=None):
    """Log the trace of the program"""
    def decorator_tracker(func)	:
        @functools.wraps(func)
        def wrapper_logger(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time

            effective_logger = ulogger if ulogger is not None else logger
            effective_logger.debug('tracker', extra={
                                   **kwargs, 'function': func.__name__, 'value': value,  'duration': run_time})
            return value

        return wrapper_logger

    if _func is None:
        return decorator_tracker
    else:
        return decorator_tracker(_func)
