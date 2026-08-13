"""Shared CLI helpers for the training entrypoints.

Provides the DEPRECATED_FLAGS machinery: a declarative registry mapping CLI
flags that belong to rejected / null-result experiments to a warning printed
when a user explicitly sets them. The flags still work (kept for
reproducibility of the published experiments); the warning just makes the
"documented failed experiment" status visible at the point of use.
"""
import sys
from typing import NamedTuple

from argparse import ArgumentParser, Namespace


class DeprecatedFlag(NamedTuple):
    """A CLI flag that is retained only for reproducing a closed experiment.

    flag:   user-facing argument spelled on the command line (e.g. "--lq").
    dest:   the argparse attribute it populates.
    status: short verdict label (e.g. "null result (Exp 12)", "rejected (Exp 13)").
    note:   one/two sentence summary of why it should not be used.
    """

    flag: str
    dest: str
    status: str
    note: str

    def is_used(self, args: Namespace, parser: ArgumentParser) -> bool:
        """True only when the user explicitly changed the flag from its default.

        Store-true flags warn when True; value flags warn when they differ from
        the argparse default, so passing no value never triggers a warning.
        """
        return getattr(args, self.dest) != parser.get_default(self.dest)


def warn_deprecated(parser: ArgumentParser, args: Namespace,
                    flags: list[DeprecatedFlag]) -> int:
    """Print a warning for every deprecated flag the user explicitly set.

    Warnings go to stderr so stdout stays machine-readable. Returns the number
    of flags warned about (0 when nothing deprecated was requested).
    """
    used = [f for f in flags if f.is_used(args, parser)]
    if not used:
        return 0
    print("=" * 72, file=sys.stderr)
    print(f"DEPRECATED FLAGS USED ({len(used)}):", file=sys.stderr)
    for f in used:
        print(f"  {f.flag}  [{f.status}]", file=sys.stderr)
        print(f"    {f.note}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    return len(used)