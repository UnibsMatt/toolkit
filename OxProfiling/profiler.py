import cProfile
import pstats

from line_profiler import LineProfiler


def profile(fnc):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        return_value = fnc(*args, **kwargs)
        pr.disable()
        sort_by = "cumulative"
        ps = pstats.Stats(pr).strip_dirs().sort_stats(sort_by)
        ps.print_stats()
        return return_value

    return inner


def line_profile(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()

    return wrapper
