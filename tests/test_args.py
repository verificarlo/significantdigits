import pytest

from significantdigits.args import parse_args

default_inputs_args = ["--inputs", "[1,2,3]", "--input-format", "stdin"]
default_metric_args = ["--metric=Significant"]


def test_args_inputs_stdin():
    with pytest.raises(SystemExit):
        parse_args([*default_metric_args, "--inputs ", "--input-format", "stdin"])
    with pytest.raises(SystemExit):
        parse_args(
            [*default_metric_args, "--inputs", "[1,a,3]", "--input-format", "stdin"]
        )
    parse_args(
        [*default_metric_args, "--inputs", "[1,2.0,3.0]", "--input-format", "stdin"]
    )


def test_args_metric(parser):
    with pytest.raises(SystemExit):
        parse_args(["--metric", *default_inputs_args])

    parse_args(["--metric=Significant", *default_inputs_args])
    parse_args(["--metric=Contributing", *default_inputs_args])


def test_args_method(parser):
    with pytest.raises(SystemExit):
        parse_args(["--method", *default_metric_args, *default_inputs_args])

    parse_args(["--method=CNH", *default_metric_args, *default_inputs_args])
    parse_args(["--method=General", *default_metric_args, *default_inputs_args])


def test_args_probability(parser):
    with pytest.raises(TypeError):
        parse_args(["--probability=-1", *default_metric_args, *default_inputs_args])

    with pytest.raises(TypeError):
        parse_args(["--probability=2", *default_metric_args, *default_inputs_args])

    parse_args(["--probability=0", *default_metric_args, *default_inputs_args])
    parse_args(["--probability=1", *default_metric_args, *default_inputs_args])


def test_args_confidence(parser):
    with pytest.raises(TypeError):
        parse_args(["--confidence=-1", *default_metric_args, *default_inputs_args])

    with pytest.raises(TypeError):
        parse_args(["--confidence=2", *default_metric_args, *default_inputs_args])

    parse_args(["--confidence=0", *default_metric_args, *default_inputs_args])
    parse_args(["--confidence=1", *default_metric_args, *default_inputs_args])
