import argparse
import os
import sys
import time

from src.yos_pupil_detection import get_centroid

my_parser = argparse.ArgumentParser(description='Pupil centroid finder function. Input file path to video file. Use '
                                                '"spacebar" to pause/play. While paused, use "k" to go to next frame')
my_parser.add_argument('-p', '--Path', type=str, metavar='', help='File path to video file', required=True)
my_parser.add_argument('-s', '--smooth', action='store_true', help='Smooth mode')
my_parser.add_argument('-d', '--display', action='store_true', help='Display layer view')

args = my_parser.parse_args()

input_path = args.Path

if not os.path.exists(input_path):
    print('The file specified does not exist')
    sys.exit()

if __name__ == '__main__':
    start = time.process_time()
    centroids = get_centroid(input_path, args.smooth, args.display)
    print(centroids)
    print(time.process_time()-start, len(centroids))