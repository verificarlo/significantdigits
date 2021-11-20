from significantdigits.stats.dispatch import dispatch


def mean(a, /, *args, **kwargs):
    _mean = dispatch(a, 'mean')
    return _mean(a, *args, **kwargs)


def var(a, /, *args, **kwargs):
    _var = dispatch(a, 'var')
    return _var(a, *args, **kwargs)


def std(a, /, *args, **kwargs):
    _std = dispatch(a, 'std')
    return _std(a, *args, **kwargs)


def absolute_error(x, y):
    _abs = dispatch(x, 'absolute_error')
    return _abs(x, y)


def relative_error(x, y):
    _rel = dispatch(x, 'relative_error')
    return _rel(x, y)


def asarray(x):
    _asarray = dispatch(x, 'asarray')
    return _asarray(x)
