import time

from src.pupil_detect_functions import find_centroids
import argparse

print(find_centroids("test_videos/sample.mkv", "snap"))

# find_centroids("test_videos/Video.mp4")