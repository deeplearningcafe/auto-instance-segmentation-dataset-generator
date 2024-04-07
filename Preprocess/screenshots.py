import cv2
import os
from tqdm import tqdm

from utils.logs import create_logger

log = create_logger(__name__)


def get_frames(video_path, output_path, time_lapse, video_name):
    """Saves frames from a video

    Args:
        video_path (_type_): video file path
        output_path (_type_): output folder
        time_lapse (_type_): how many frames has to pass before saving.
            if time_lapse = 10 then every 10 frames save
        video_name (_type_): the name to save the frames:
            {video_name}_frame_{count}
    """
    
    #creating a folder
    try:  
        # creating a folder named data 
        if not os.path.exists(output_path): 
            os.makedirs(output_path)

    except OSError:
        log.error('Error! Could not create a directory') 

    # open the video
    cam = cv2.VideoCapture(video_path)

    # read fps
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cam.get(cv2.CAP_PROP_FRAME_COUNT)

    current_frame = 0
    count = 0
    log.info(f"The fps of the video is: {frames_per_second}")
    # time_lapse = int(frames_per_second/2)
    log.info(f"The number of frames to save is: {time_lapse}")
    progress_bar = iter(tqdm(range(int(totalNoFrames+1))))

    while True:
        # read one frame, ret = 画像取得が成功した場合の表示
        ret, frame = cam.read()
        next(progress_bar)

        if ret:
            if current_frame%time_lapse == 0: 
                save_path = os.path.join(output_path, f"{video_name}_frame_{count}.jpg")
                cv2.imwrite(save_path, frame)
                count += 1
            
            current_frame += 1

        if not ret:
            log.info("No frame")
            break
    
    # we could add here one next(progress_bar) to end the bar, test with a not so long video
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    video_path = r""
    output_path = ""
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    time_lapse = 24
    video_name = ""
    
    get_frames(video_path, output_path, time_lapse, video_name)