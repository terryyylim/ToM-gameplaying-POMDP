from typing import Any

import os
import bz2
import cv2
import datetime
import pickle
from pathlib import Path

def make_video_from_image_dir(vid_path, img_folder, video_name='trajectory', fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)

def make_video_from_rgb_imgs(rgb_arrs, vid_path, video_name='trajectory',
                             fps=5, format="mp4v", resize=(640, 480)):
    """
    Create a video from a list of rgb arrays
    """
    # print("Rendering video...")
    if vid_path[-1] != '/':
        vid_path += '/'
    video_path = vid_path + video_name + '.mp4'

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        # if percent_done % 20 == 0:
            # print("\t...", percent_done, "% of frames rendered")
        if resize is not None:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        video.write(image)

    video.release()
    cv2.destroyAllWindows()

def save(data: Any) -> None:
    persistence_path = 'episodes_cache'
    if not os.path.isdir(persistence_path):
        os.mkdir(persistence_path)

    pickle_file = Path(f'episodes_{datetime.datetime.utcnow()}.pbz2')
    pickle_file = (persistence_path / pickle_file).as_posix()

    handler = bz2.BZ2File(pickle_file, 'w')
    with handler as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print(f'Episodes cache saved to {pickle_file}')
