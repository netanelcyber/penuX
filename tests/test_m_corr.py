import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m import _effective_corr_every


def test_effective_corr_every_small_epochs_default():
    assert _effective_corr_every(100, 1) == 1


def test_effective_corr_every_large_epochs_default():
    assert _effective_corr_every(1000, 1) == 10


def test_effective_corr_every_large_epochs_custom():
    # user-specified non-default should be respected
    assert _effective_corr_every(1000, 5) == 5


def test_effective_corr_every_zero_off():
    # 0 turns off saving
    assert _effective_corr_every(2000, 0) == 0


def test_effective_corr_every_force():
    # force should honor the user setting
    assert _effective_corr_every(1000, 1, force=True) == 1
