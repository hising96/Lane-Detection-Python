#code refernced from stackOverflow

import cv2

# Enter the location of the video - add folders if not in current directory
video_location = 'C:/Users/HIMANSHU/Desktop/delhi-gurgaon.mp4'

vidcap = cv2.VideoCapture(video_location)
success, image = vidcap.read()
count = 0
success = True

# Iterates through all video frames until it runs out (i.e. video ends)
# Change for desired location to save image files extracted
# If putting in a folder, folder must have already been created
while success:
    success, image = vidcap.read()
    cv2.imwrite('my_vid_1/frame%d.jpg' % count, image)
    count += 1