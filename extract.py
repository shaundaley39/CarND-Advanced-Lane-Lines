#!/usr/bin/env python3

import os
from moviepy.editor import *
import glob


# movie_glob = '*.mp4'
# imgdir = 'video_images'
# movie_glob = 'project_video.mp4'
movie_glob = 'project*.mp4'
imgdir = 'project_images'
N = 200


def extract_frames(movie, imgdir):
    clip = VideoFileClip(movie)
    times = [(i*(clip.duration/N)) for i in range(N)]
    for t in times:
        imgpath = os.path.join(imgdir, movie[:-4] + '-{}.jpg'.format(t))
        clip.save_frame(imgpath, t)


movies = glob.glob(movie_glob)
for movie in movies:
    extract_frames(movie, imgdir)
