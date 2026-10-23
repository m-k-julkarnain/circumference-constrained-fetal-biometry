"""Shared CLI plumbing for the scripts."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fetalbio import config as C  # noqa: E402


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data-root", default=C.DATA_ROOT,
                   help="folder containing annotations/ and images/ (or set FETALBIO_DATA)")
    p.add_argument("--out", default="outputs", help="output folder (results/ and checkpoints/)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(C.SEEDS))
    return p


def parse(p):
    a = p.parse_args()
    C.set_data_root(a.data_root)
    os.makedirs(a.out, exist_ok=True)
    return a
