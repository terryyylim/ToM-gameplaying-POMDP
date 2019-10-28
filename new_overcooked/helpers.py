from typing import List

import os
import cv2
import glob

def check_dir_exist(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def clean_dir(dir_path: str) -> None:
    for file in os.listdir(dir_path):
        # CASE: Images directory
        if dir_path == 'new_overcooked/simulations':
            if file.endswith('.png'):
                os.remove(dir_path+'/'+file)

def get_video_count(dir_path):
    game_folder = os.path.dirname(__file__)
    video_folder = os.path.join(game_folder, 'videos')
    video_count = str(len(glob.glob1(video_folder, dir_path+'*.*')))
    return video_count

def get_video_name_ext(agent_type: List[bool], episodes: int) -> str:
    video_name_ext = ['ToM' if a_type else 'Dummy' for a_type in agent_type]
    video_name_ext = '_'.join(video_name_ext) + '_' + str(episodes) + '_ep'
    video_type_count = get_video_count(video_name_ext)
    return video_name_ext + '_' + video_type_count

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
