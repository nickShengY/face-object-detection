import os

import tensorflow as tf
import json
import numpy as np
from matplotlib import pyplot as plt
import cv2

# Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

def augment_data():    
    images = tf.data.Dataset.list_files('data\\images\\*.jpg')
    images.as_numpy_iterator().next()
    def load_image(x): 
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img

    images = images.map(load_image)

    images.as_numpy_iterator().next()

    print("Type of images", type(images))

    image_generator = images.batch(4).as_numpy_iterator()

    plot_images = image_generator.next()

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, image in enumerate(plot_images):
        ax[idx].imshow(image) 
    plt.show()

    for folder in ['train','test','val']:
        for file in os.listdir(os.path.join('data', folder, 'images')):
            
            filename = file.split('.')[0]+'.json'
            existing_filepath = os.path.join('data','labels', filename)
            if os.path.exists(existing_filepath): 
                new_filepath = os.path.join('data',folder,'labels',filename)
                os.replace(existing_filepath, new_filepath)      

    import albumentations as alb

    augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                            alb.HorizontalFlip(p=0.5), 
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.RandomGamma(p=0.2), 
                            alb.RGBShift(p=0.2), 
                            alb.VerticalFlip(p=0.5)], 
                        bbox_params=alb.BboxParams(format='albumentations', 
                                                    label_fields=['class_labels']))

    print("Gathering Pictures")    

    img = cv2.imread(os.path.join('data','train', 'images','12.jpg'))



    with open(os.path.join('data', 'train', 'labels', '12.json'), 'r') as f:
        label = json.load(f)


    coords = [0,0,0,0]
    coords[0] = label['shapes'][0]['points'][0][0]
    coords[1] = label['shapes'][0]['points'][0][1]
    coords[2] = label['shapes'][0]['points'][1][0]
    coords[3] = label['shapes'][0]['points'][1][1]

    print("Normalizing...")    

    coords = list(np.divide(coords, [640,480,640,480]))
    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

    cv2.rectangle(augmented['image'], 
                tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
                tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)), 
                        (255,0,0), 2)

    plt.imshow(augmented['image'])

    print("Augmenting collected data...")

    for partition in ['train','test','val']: 
        for image in os.listdir(os.path.join('data', partition, 'images')):
            img = cv2.imread(os.path.join('data', partition, 'images', image))

            coords = [0,0,0.00001,0.00001]
            label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)

                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                coords = list(np.divide(coords, [640,480,640,480]))
            label_dir = os.path.join('aug_data', partition, 'labels')
            os.makedirs(label_dir, exist_ok=True)  # create directory if not exists
            

            try: 
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                    label_dir_2 = os.path.join('aug_data', partition, 'images')
                    os.makedirs(label_dir_2, exist_ok=True)  # create directory if not exists
                    cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0: 
                            annotation['bbox'] = [0,0,0,0]
                            annotation['class'] = 0 
                        else: 
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 


                    with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation, f)


            except Exception as e:
                print(e)
    print("Success!")  
                
    print("Loading to tensorflow dataset...")            
    train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
    train_images = train_images.map(lambda x: x/255)
                
                
    test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
    test_images = test_images.map(lambda x: x/255)

    val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
    val_images = val_images.map(load_image)
    val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
    val_images = val_images.map(lambda x: x/255)

    train_images.as_numpy_iterator().next()

    def load_labels(label_path):
        with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
            label = json.load(f)
            
        return [label['class']], label['bbox']

    train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    train_labels.as_numpy_iterator().next()

    print("Matchng Labels...")

    print("checking numbers: ")

    print("train size:", len(train_images), "train labels:", len(train_labels), "test size:", len(test_images), len(test_labels),"validation size:", len(val_images), len(val_labels))

    print("Preparing data for training...")

    train = tf.data.Dataset.zip((train_images, train_labels))
    train = train.shuffle(5000)
    train = train.batch(8)
    train = train.prefetch(4)

    test = tf.data.Dataset.zip((test_images, test_labels))
    test = test.shuffle(1300)
    test = test.batch(8)
    test = test.prefetch(4)

    val = tf.data.Dataset.zip((val_images, val_labels))
    val = val.shuffle(1000)
    val = val.batch(8)
    val = val.prefetch(4)

    train.as_numpy_iterator().next()[1]

    data_samples = train.as_numpy_iterator()

    res = data_samples.next()

    print("Prepared data: ")

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx in range(4): 
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]
        
        cv2.rectangle(sample_image, 
                    tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                    tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)

        ax[idx].imshow(sample_image)
    return train, test, val

train, test, val = augment_data()
if __name__ == '__main__':
    pass
    # train, test, val = augment_data()
