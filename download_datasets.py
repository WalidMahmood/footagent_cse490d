"""
FootAgent dataset downloader.

Usage examples:
  python download_datasets.py --statsbomb
  python download_datasets.py --soccernet-tracking --splits train test
  python download_datasets.py --soccernet-tracking-2023 --splits train test
  python download_datasets.py --soccernet-videos --splits train valid test --password "<NDA_PASSWORD>"
  python download_datasets.py --all-open

Notes:
- Some SoccerNet assets require NDA/password for video files.
- MVFouls availability can vary by SoccerNet package version/task naming and access rights.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_statsbomb(data_root: Path) -> None:
    target = data_root / "statsbomb" / "open-data"
    ensure_dir(target.parent)

    if target.exists() and any(target.iterdir()):
        print(f"[skip] StatsBomb repo already exists: {target}")
        return

    run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/statsbomb/open-data.git",
        str(target),
    ])


def _get_soccernet_downloader(local_dir: Path):
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except Exception as exc:
        raise RuntimeError(
            "SoccerNet package is not installed. Run: pip install SoccerNet"
        ) from exc

    ensure_dir(local_dir)
    return SoccerNetDownloader(LocalDirectory=str(local_dir))


def download_soccernet_tracking(data_root: Path, splits: list[str]) -> None:
    downloader = _get_soccernet_downloader(data_root / "soccernet")
    print(f"[info] Downloading SoccerNet tracking split={splits}")
    downloader.downloadDataTask(task="tracking", split=splits)


def download_soccernet_tracking_2023(data_root: Path, splits: list[str]) -> None:
    downloader = _get_soccernet_downloader(data_root / "soccernet")
    print(f"[info] Downloading SoccerNet tracking-2023 split={splits}")
    downloader.downloadDataTask(task="tracking-2023", split=splits)


def download_soccernet_videos(data_root: Path, splits: list[str], password: str | None) -> None:
    downloader = _get_soccernet_downloader(data_root / "soccernet")
    if password:
        downloader.password = password

    print(f"[info] Downloading SoccerNet 720p videos split={splits}")
    downloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=splits)


def download_soccernet_mvfouls_hint(data_root: Path) -> None:
    ensure_dir(data_root / "soccernet" / "mvfouls")
    print("[todo] MVFouls download requires task/access confirmation in your SoccerNet account.")
    print("[todo] See MANUAL_DATASET_LINKS.md for official links and workflow.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FootAgent datasets")
    parser.add_argument("--data-root", default="data", help="Data root folder")

    parser.add_argument("--statsbomb", action="store_true", help="Download StatsBomb open-data repo")
    parser.add_argument("--soccernet-tracking", action="store_true", help="Download SoccerNet tracking task")
    parser.add_argument("--soccernet-tracking-2023", action="store_true", help="Download SoccerNet tracking-2023 task")
    parser.add_argument("--soccernet-videos", action="store_true", help="Download SoccerNet 720p videos (NDA/password may be required)")
    parser.add_argument("--soccernet-mvfouls", action="store_true", help="Create MVFouls target dir and print access instructions")

    parser.add_argument("--all-open", action="store_true", help="Download only open-access assets")
    parser.add_argument("--splits", nargs="+", default=["test"], help="SoccerNet splits")
    parser.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"), help="SoccerNet password for protected files")

    args = parser.parse_args()
    data_root = Path(args.data_root)

    if args.all_open:
        download_statsbomb(data_root)
        download_soccernet_tracking(data_root, splits=args.splits)
        download_soccernet_tracking_2023(data_root, splits=args.splits)
        return

    ran = False

    if args.statsbomb:
        download_statsbomb(data_root)
        ran = True

    if args.soccernet_tracking:
        download_soccernet_tracking(data_root, splits=args.splits)
        ran = True

    if args.soccernet_tracking_2023:
        download_soccernet_tracking_2023(data_root, splits=args.splits)
        ran = True

    if args.soccernet_videos:
        download_soccernet_videos(data_root, splits=args.splits, password=args.password)
        ran = True

    if args.soccernet_mvfouls:
        download_soccernet_mvfouls_hint(data_root)
        ran = True

    if not ran:
        parser.print_help()


if __name__ == "__main__":
    main()
