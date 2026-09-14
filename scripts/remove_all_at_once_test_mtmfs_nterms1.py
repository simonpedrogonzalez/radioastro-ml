"""Run the one-step removal test with MT-MFS and one Taylor term.

This is the MT-MFS counterpart of ``remove_all_at_once_test.py``.  It keeps
the same simulated dataset, imaging geometry, one-pixel mask, ``niter=1``, and
``gain=1.0``, while explicitly selecting:

* deconvolver="mtmfs"
* nterms=1
* weighting="briggs"
* robust=0.5

Run inside CASA from the repository root with::

    execfile('scripts/remove_all_at_once_test_mtmfs_nterms1.py')
"""

from pathlib import Path

from scripts import remove_all_at_once_test as test


test.DECONVOLVER = "mtmfs"
test.NTERMS = 1
test.WEIGHTING = "briggs"
test.ROBUST = 0.5
test.OUTPUT_ROOT = (
    Path(test.REPO_ROOT) / "experiments" / "remove_all_at_once_test_07_mtmfs_nterms1"
)


def main() -> dict:
    return test.main()


main()
