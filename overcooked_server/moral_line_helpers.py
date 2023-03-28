from typing import List  # not used
from typing import Union  # not used

# importing with "as" allows you to easily change files if you update the libraries
import os as os
import numpy
import cv2 as cv  # changed requirements to require more recent version of openCV as the last one was causing errors.
import glob  # not quite sure why this is imported as its not used.


def check_dir_exist(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def clean_dir(dir_path: str) -> None:
    for file in os.listdir(dir_path):
        # CASE: Images directory
        if dir_path == 'overcooked_server/simulations':
            if file.endswith('.png'):
                os.remove(dir_path + '/' + file)


def make_video_from_rgb_imgs(rgb_arrs, vid_path, video_name='trajectory',
                             fps=1, format="mp4v", resize=(1008, 1080)):
    """
    Create a video from a list of rgb arrays
    """
    # print("Rendering video...")
    if vid_path[-1] != '/':
        vid_path += '/'
    video_path = vid_path + video_name + '.mp4'
    fourcc = cv.VideoWriter_fourcc(*format)
    video = cv.VideoWriter(video_path, fourcc, 20, resize) # to low of a frame rate will cause imporper rendering.
    for image in rgb_arrs:
        # percent_done = int((i / len(rgb_arrs)) * 100)
        # if percent_done % 20 == 0:
        #     print("\t...", percent_done, "% of frames rendered")

        # had to change frame rate to allow for proper rendering
        for i in range(20//fps):
            if resize is not None:
                num_im = numpy.array(image)
                image = cv.resize(num_im, resize, interpolation=cv.INTER_AREA)
                video.write(image)
    video.release()
    cv.destroyAllWindows()


def make_video_from_image_dir(vid_path, img_folder, video_name='trajectory', fps=1):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)
    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)
