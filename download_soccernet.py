"""
Download SoccerNet tracking dataset clips for testing.
Downloads a small subset of the tracking dataset for validation.
"""

from SoccerNet.Downloader import SoccerNetDownloader
import os

# Create local directory
download_dir = "data/soccernet"
os.makedirs(download_dir, exist_ok=True)

# Initialize downloader
sn_downloader = SoccerNetDownloader(LocalDirectory=download_dir)

# Download tracking dataset (this may take a while)
print("Downloading SoccerNet tracking dataset...")
sn_downloader.downloadDataTask(task="tracking", split=["test"])

print("Download complete! Check data/soccernet for clips.")
