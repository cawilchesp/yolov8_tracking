import cv2
import glob

source_folder = 'C:/Users/cawil/OneDrive/Escritorio/DEEP_CCTV/'
source_file = '20230412_CCTV_Barrio_Triste'
image_files = sorted(glob.glob(f'{source_folder}{source_file}/*.png'))

output_file = f'{source_folder}output_{source_file}.mp4'
frame_rate = 20.0
video_size = (1280, 720)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, video_size)

for image_file in image_files:
    img = cv2.imread(image_file)
    video_writer.write(img)

video_writer.release()
cv2.destroyAllWindows()
