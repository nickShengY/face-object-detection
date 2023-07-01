import os
import cv2
import time
from config import IMAGES_PATH, NUM_IMAGES

def collect_data():
    # Create the IMAGES_PATH if it does not already exist
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    
    # Start the webcam capture
    cap = cv2.VideoCapture(0)

    for imgnum in range(NUM_IMAGES):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        
        # Ensure the frame was read successfully
        if ret:
            imgname = os.path.join(IMAGES_PATH, f'{str(imgnum)}.jpg')
            cv2.imwrite(imgname, frame)
            cv2.imshow('frame', frame)
            time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Error capturing image')
    
    # Close the camera and any open windows
    cap.release()
    cv2.destroyAllWindows()
    print("Please label your collected face data. \n \tCaution: Please only use Rectangle and place them in the right dir!")
    os.system('labelme')


if __name__ == '__main__':
    pass
    # collect_data()
    

