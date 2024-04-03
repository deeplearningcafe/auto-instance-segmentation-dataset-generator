import cv2
import os


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
        print ('Error! Could not create a directory') 

    # open the video
    cam = cv2.VideoCapture(video_path)

    # read fps
    frames_per_second = cam.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    count = 0
    print("The fps of the video is:", frames_per_second)
    # time_lapse = int(frames_per_second/2)
    print("The number of frames to save is: ", time_lapse)
    while True:
        # read one frame, ret = 画像取得が成功した場合の表示
        ret, frame = cam.read()

        if ret:
            if current_frame%time_lapse == 0: 
                save_path = os.path.join(output_path, f"{video_name}_frame_{count}.jpg")
                cv2.imwrite(save_path, frame)
                count += 1
            
            current_frame += 1

        if not ret:
            print("No frame")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    video_path = r"G:\アニメ\アキバ\[UCCUSS] Akiba Meido Sensou アキバ冥途戦争 第2巻 (BD 1920x1080p AVC FLAC)\[UCCUSS] Akiba Meido Sensou アキバ冥途戦争 第05話 「赤に沈む! 三十六歳生誕祭!」 (BD 1920x1080p AVC FLAC).mkv"
    output_path = "e:\Data\object_dection\maids"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    time_lapse = 24
    video_name = "chap_5"
    
    get_frames(video_path, output_path, time_lapse, video_name)