"""Resume an Icentia11k download without re-indexing.

    python resume_icentia.py --root E:\\munji\\raw --check     # report only
    python resume_icentia.py --root E:\\munji\\raw
    python resume_icentia.py --root E:\\munji\\raw --patients 300

wfdb.dl_database asks PhysioNet to enumerate every record before transferring
anything, and repeats that on every retry. With 2,000 patients that is several
minutes of nothing per restart.

Icentia's layout is regular, so the file list is derived locally instead:

    p00/p00000/p00000_s00.hea|.dat|.atr
    ...
    p01/p01999/p01999_s49.hea|.dat|.atr

Not every file in that grid exists. Some segments carry no annotation file, and
a few patients have fewer than 50 segments. Those gaps are confirmed once and
recorded in `.absent`, so a later run does not re-request them — without that,
a patient missing one .atr looks permanently incomplete and gets re-probed on
every run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BASE = "https://physionet.org/files/icentia11k-continuous-ecg/1.0"

# .hea and .dat are always present. The annotation extension varies by dataset
# (.atr, .ecg, .qrs), so it is probed rather than assumed.
CORE_EXTS = (".hea", ".dat")
ANN_CANDIDATES = (".atr", ".ecg", ".qrs")

MAX_SEGMENTS = 50
RETRIES = 4
BACKOFF = 5
ABSENT_FILE = ".absent"


def patient_dir(root: Path, pid: int) -> Path:
    return Path(root) / "icentia11k" / f"p{pid // 1000:02d}" / f"p{pid:05d}"


def remote_url(pid: int, seg: int, ext: str) -> str:
    return f"{BASE}/p{pid // 1000:02d}/p{pid:05d}/p{pid:05d}_s{seg:02d}{ext}"


def rel_name(pid: int, seg: int, ext: str) -> str:
    return f"p{pid:05d}_s{seg:02d}{ext}"


# ------------------------------------------------------------------ absent set


def absent_path(root: Path) -> Path:
    return Path(root) / "icentia11k" / ABSENT_FILE


def load_absent(root: Path) -> set[str]:
    p = absent_path(root)
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}


def record_absent(root: Path, name: str) -> None:
    """Append immediately so the record survives Ctrl+C mid-run."""
    p = absent_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(name + "\n")


# ------------------------------------------------------------------ discovery


def probe_annotation_ext(root: Path) -> str:
    """Which annotation extension this dataset uses.

    Local files answer it for free; only a fresh download needs the network.
    """
    base = Path(root) / "icentia11k"
    if base.exists():
        for ext in ANN_CANDIDATES:
            if next(base.rglob(f"*{ext}"), None) is not None:
                return ext
    for ext in ANN_CANDIDATES:
        try:
            with urlopen(remote_url(0, 0, ext), timeout=30):
                return ext
        except HTTPError as e:
            if e.code != 404:
                raise
        except (URLError, TimeoutError, OSError):
            break
    return ".atr"


def missing_files(root: Path, pid: int, exts: tuple[str, ...],
                  absent: set[str]) -> list[tuple[int, str]]:
    """Files not on disk and not already confirmed absent on the server."""
    d = patient_dir(root, pid)
    out = []
    for s in range(MAX_SEGMENTS):
        for e in exts:
            name = rel_name(pid, s, e)
            if name in absent:
                continue
            f = d / name
            # Zero bytes means a failed transfer, not a completed one.
            if not f.exists() or f.stat().st_size == 0:
                out.append((s, e))
    return out


def scan(root: Path, patients: int, exts: tuple[str, ...],
         absent: set[str]) -> tuple[list[int], list[int], int]:
    complete, incomplete, files = [], [], 0
    for pid in range(patients):
        d = patient_dir(root, pid)
        files += len(list(d.glob("*.hea"))) if d.exists() else 0
        (complete if not missing_files(root, pid, exts, absent)
         else incomplete).append(pid)
    return complete, incomplete, files


# ------------------------------------------------------------------- transfer


def fetch(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download one file. False means a confirmed 404; errors raise.

    Writes to a temporary name and renames on success, so an interrupted
    transfer cannot leave a truncated file that a later run treats as done.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = BACKOFF
    for attempt in range(RETRIES):
        try:
            with urlopen(url, timeout=timeout) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except HTTPError as e:
            if e.code == 404:
                return False
            if attempt == RETRIES - 1:
                raise
        except (URLError, TimeoutError, OSError):
            if attempt == RETRIES - 1:
                raise
        time.sleep(delay)
        delay *= 2
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--patients", type=int, default=2000)
    ap.add_argument("--check", action="store_true", help="report and exit")
    a = ap.parse_args(argv)

    ann = probe_annotation_ext(a.root)
    exts = CORE_EXTS + (ann,)
    absent = load_absent(a.root)

    print(f"annotation extension : {ann}")
    print(f"known absent files   : {len(absent):,}")
    print(f"scanning for {a.patients:,} patients")
    complete, incomplete, files = scan(a.root, a.patients, exts, absent)
    print(f"  complete   {len(complete):,}")
    print(f"  remaining  {len(incomplete):,}")
    print(f"  .hea files {files:,}")

    if a.check:
        return 0
    if not incomplete:
        print("\nnothing to do — all requested patients are complete")
        return 0

    # Finish partial patients first: one with 49 of 50 segments is useless to
    # the pipeline, so completing it beats starting a new one.
    incomplete.sort(key=lambda p: -(len(list(patient_dir(a.root, p).glob("*.hea")))
                                    if patient_dir(a.root, p).exists() else 0))

    print("\ndownloading — Ctrl+C is safe, progress is kept\n")
    done = ok = gone = 0
    t0 = time.time()

    try:
        for pid in incomplete:
            todo = missing_files(a.root, pid, exts, absent)
            if not todo:
                continue
            d = patient_dir(a.root, pid)
            got = miss = 0
            for seg, ext in todo:
                name = rel_name(pid, seg, ext)
                try:
                    if fetch(remote_url(pid, seg, ext), d / name):
                        ok += 1
                        got += 1
                    else:
                        absent.add(name)
                        record_absent(a.root, name)
                        gone += 1
                        miss += 1
                except Exception as e:
                    print(f"\n[fail] {name}: {type(e).__name__}: {e}")
                    print("rerun to continue where it stopped")
                    return 1
            done += 1
            rate = ok / max(time.time() - t0, 1) * 60
            tag = f"{got} new" + (f", {miss} absent" if miss else "")
            print(f"  p{pid:05d}  ({done}/{len(incomplete)})  {tag}  "
                  f"| {ok:,} files, {rate:.0f}/min, ~{len(incomplete)-done} left",
                  flush=True)
    except KeyboardInterrupt:
        print("\n\nstopped. rerun to continue — nothing is lost.")
        return 0

    print(f"\ndone: {ok:,} downloaded, {gone:,} absent on server, "
          f"{done:,} patients finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
