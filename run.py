import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set visible GPUs to -1 (none) if you do not want to use GPU

from data_collection import collect_data
import cv2


def run(commands):

    for command in commands:
        print("Processing: ", command)
        if command == 'collect_data':
            collect_data()
            print("Please Split your Training, Test, val images manually")

        elif command == 'augment_data':
            print("Start Augmenting...")
            from data_augmentation import augment_data
            augment_data()

        elif command == 'train_model':
            from model import create_model
            create_model()
            from training import train_model
            train_model()

        elif command == 'live_detection':
            from live_detection import live_detection, cap
            live_detection()

        elif command == 'Force_camera_off':
            from live_detection import live_detection, cap
            cap.release()
            cv2.destroyAllWindows()

        else:
            print(f'Unknown command {command}')


if __name__ == "__main__":

    commands = sys.argv[1:]
    run(commands)
