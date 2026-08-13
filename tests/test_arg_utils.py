"""Tests for the shared deprecated-flag registry (ternary_llm/arg_utils.py)."""
import argparse

from ternary_llm.arg_utils import DeprecatedFlag, warn_deprecated


def _make_parser(store_true: list, values: list):
    p = argparse.ArgumentParser()
    for name in store_true:
        p.add_argument("--" + name, action="store_true", default=False)
    for name, default in values:
        p.add_argument("--" + name, type=float, default=default)
    return p


def test_store_true_deprecated_only_warns_when_explicit(capsys):
    parser = _make_parser(store_true=["sr"], values=[])
    flags = [DeprecatedFlag("--sr", "sr", "rejected (Exp 13)", "note")]

    args = parser.parse_args([])  # not set
    assert warn_deprecated(parser, args, flags) == 0
    assert capsys.readouterr().err == ""

    args = parser.parse_args(["--sr"])  # explicitly set
    assert warn_deprecated(parser, args, flags) == 1
    assert "--sr" in capsys.readouterr().err


def test_value_deprecated_warns_only_when_changed(capsys):
    parser = _make_parser(store_true=[], values=[("v8-reg", 0.0)])
    flags = [DeprecatedFlag("--v8-reg", "v8_reg", "rejected (Exp 11)", "note")]

    args = parser.parse_args([])  # default = no warning
    assert warn_deprecated(parser, args, flags) == 0

    args = parser.parse_args(["--v8-reg", "5.0"])  # changed = warning
    assert warn_deprecated(parser, args, flags) == 1


def test_multiple_and_zero(capsys):
    parser = _make_parser(store_true=["lq", "sr"], values=[])
    flags = [
        DeprecatedFlag("--lq", "lq", "null result (Exp 12)", "n"),
        DeprecatedFlag("--sr", "sr", "rejected (Exp 13)", "n"),
    ]
    args = parser.parse_args(["--lq", "--sr"])
    assert warn_deprecated(parser, args, flags) == 2
    err = capsys.readouterr().err
    assert "--lq" in err and "--sr" in err
