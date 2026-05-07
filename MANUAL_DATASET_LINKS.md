# FootAgent Dataset Links and Access

## 1) StatsBomb 360 Open Data (Open)
- Official repo: https://github.com/statsbomb/open-data
- 360 files location in repo: data/three-sixty
- Local target folder: data/statsbomb/open-data

## 2) SoccerNet-Tracking (Large, mostly open metadata/tasks)
- Tracking task docs: https://github.com/SoccerNet/sn-tracking
- SoccerNet package docs: https://github.com/SoccerNet/SoccerNet
- Local target folder: data/soccernet/tracking
- Python API examples:
  - downloadDataTask(task="tracking", split=["train","test","challenge"])
  - downloadDataTask(task="tracking-2023", split=["train","test","challenge"])

## 3) SoccerNet-MVFouls (Access-controlled)
- SoccerNet homepage: https://www.soccer-net.org/
- SoccerNet package: https://github.com/SoccerNet/SoccerNet
- Local target folder: data/soccernet/mvfouls
- Notes:
  - Access rules may require registration/NDA for some assets.
  - Use your approved credentials/password in SoccerNet downloader when requested.

## 4) SoccerNet-v3 / Videos (Usually NDA/password for full videos)
- SoccerNet package: https://github.com/SoccerNet/SoccerNet
- Local target folder: data/soccernet/videos
- Notes:
  - Video files can require password from NDA workflow.

## 5) Optional Datasets
- Metrica sample data: https://github.com/metrica-sports/sample-data
- Roboflow football datasets: https://universe.roboflow.com/

## Automated script in this repo
- Script: download_datasets.py
- Open assets example:
  python download_datasets.py --all-open --splits test
- StatsBomb only:
  python download_datasets.py --statsbomb
- Tracking only:
  python download_datasets.py --soccernet-tracking --splits test
