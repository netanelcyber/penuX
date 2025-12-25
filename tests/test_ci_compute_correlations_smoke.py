import os
import shutil
import subprocess
import sys
from pathlib import Path

def test_compute_correlations_smoke(tmp_path):
    # prepare tiny CSV
    csv = tmp_path / "clinical_small.csv"
    csv.write_text("""temperature_c,wbc,spo2,age,label
36.6,7000,98,45,Normal
38.2,15000,92,68,Bacterial
37.0,9000,95,30,Normal
39.0,20000,88,77,Viral
""")

    outdir = tmp_path / "out"
    outdir.mkdir()

    # run the module
    cmd = [sys.executable, "-m", "scripts.compute_correlations", "--input", str(csv), "--outdir", str(outdir), "--prefix", "smoke"]
    subprocess.check_call(cmd)

    # check expected files
    pear = outdir / "smoke_pearson.csv"
    spear = outdir / "smoke_spearman.csv"
    pimg = outdir / "smoke_pearson.png"
    simg = outdir / "smoke_spearman.png"

    assert pear.exists(), f"Missing {pear}"
    assert spear.exists(), f"Missing {spear}"
    assert pimg.exists(), f"Missing {pimg}"
    assert simg.exists(), f"Missing {simg}"

    # cleanup (not strictly necessary with tmp_path)
    shutil.rmtree(outdir)
