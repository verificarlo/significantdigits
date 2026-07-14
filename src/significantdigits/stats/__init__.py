from significantdigits.stats.dispatch import dispatch


def mean(a, /, *args, **kwargs):
    _mean = dispatch(a, method='mean')
    return _mean(a, *args, **kwargs)


def var(a, /, *args, **kwargs):
    _var = dispatch(a, method='var')
    return _var(a, *args, **kwargs)


def std(a, /, *args, **kwargs):
    _std = dispatch(a, method='std')
    return _std(a, *args, **kwargs)


def absolute_error(x, y):
    _abs = dispatch(x, y, method='absolute_error')
    return _abs(x, y)


def relative_error(x, y):
    _rel = dispatch(x, y, method='relative_error')
    return _rel(x, y)


def asarray(x):
    _asarray = dispatch(x, method='asarray')
    return _asarray(x)
