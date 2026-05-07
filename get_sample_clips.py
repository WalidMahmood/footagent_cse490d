"""
Download a few sample SoccerNet videos for testing.
Uses YouTube downloads since full dataset is too large.
"""

import os
import subprocess
import sys

def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_sample_clips():
    """Download sample football clips for testing."""
    
    # Create clips directory
    os.makedirs('data/matches', exist_ok=True)
    
    if not check_ytdlp():
        print("yt-dlp not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], check=True)
    
    # Sample SoccerNet YouTube clips (short highlights)
    # These are public highlight clips that SoccerNet uses
    sample_urls = [
        # You can replace these with any football highlight YouTube URLs
        # For now, let's use a public domain football clip
    ]
    
    print("No specific URLs provided. Please provide a football video URL or file path.")
    print("\nOptions:")
    print("1. Provide a YouTube URL for a football clip")
    print("2. Provide a local video file path")
    print("3. I'll create a test video from your webcam")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == '1':
        url = input("Enter YouTube URL: ").strip()
        subprocess.run([
            'yt-dlp',
            '-f', 'best[height<=720]',
            '-o', 'data/matches/test_clip.%(ext)s',
            url
        ], check=True)
        
    elif choice == '2':
        path = input("Enter local video file path: ").strip()
        if os.path.exists(path):
            import shutil
            shutil.copy(path, 'data/matches/test_clip.mp4')
            print(f"Copied {path} to data/matches/test_clip.mp4")
        else:
            print(f"File not found: {path}")
            
    elif choice == '3':
        print("\nRecording 10 seconds from webcam...")
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('data/matches/test_clip.mp4', fourcc, 25.0, (640, 480))
        
        import time
        start = time.time()
        while time.time() - start < 10:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                cv2.imshow('Recording...', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Saved test_clip.mp4")
    
    else:
        print("Invalid choice. Please run again with a valid option.")

if __name__ == '__main__':
    download_sample_clips()
