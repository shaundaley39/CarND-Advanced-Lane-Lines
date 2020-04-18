#!/usr/bin/env python3

import os
from moviepy.editor import *
import glob


movie_glob = '*.mp4'
imgdir = 'video_images'


def extract_frames(movie, imgdir):
    clip = VideoFileClip(movie)
    times = [int(i*(clip.duration/10)) for i in range(10)]
    for t in times:
        imgpath = os.path.join(imgdir, movie[:-4] + '-{}.jpg'.format(t))
        clip.save_frame(imgpath, t)


movies = glob.glob(movie_glob)
for movie in movies:
    extract_frames(movie, imgdir)
