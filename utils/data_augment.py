import copy
import numpy as np
import tensorflow as tf

def augment(img_data, C, augment=True): # cv2 -> tf.image
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data
    pool_size = 16

    img_data_aug = copy.deepcopy(img_data)
    img = tf.io.read_file(img_data['filepath'])
    img = tf.image.decode_image(img)

    if len(img.shape) == 2 or img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)

    if img.shape[0] % pool_size > 0 or img.shape[1] % pool_size > 0:
        pad_h = (pool_size - (img.shape[0] % pool_size)) % pool_size
        pad_w = (pool_size - (img.shape[1] % pool_size)) % pool_size

        img = tf.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), "CONSTANT", constant_values=0)
    print('\n', img_data['filepath'].split('\\')[-1], img.shape)

    img = tf.cast(img, dtype=tf.float32)
    

    if augment:
        rows, cols = img.shape[:2]

        if C.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = tf.image.flip_left_right(img)
            
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if C.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = tf.image.flip_up_down(img)
            
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if C.rot_90 and np.random.randint(0, 2) == 0:
            angle = np.random.randint(1, 4)
            img = tf.image.rot90(img, angle)

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 3:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 2:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                else:
                # elif angle == 1:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
        
        # if C.use_random_brightness and np.random.randint(0, 2) == 0:
        #     img = tf.image.random_brightness(img, 10)
        #     img = tf.clip_by_value(img, 0.0, 255.0)

    img_data_aug['height'] = img.shape[0]
    img_data_aug['width'] = img.shape[1]
    return img_data_aug, img

# def augment(img_data, C, augment=True):
# 	assert 'filepath' in img_data
# 	assert 'bboxes' in img_data
# 	assert 'width' in img_data
# 	assert 'height' in img_data

# 	img_data_aug = copy.deepcopy(img_data)

# 	img = cv2.imread(img_data_aug['filepath'])

# 	if augment:
# 		rows, cols = img.shape[:2]

# 		if C.use_horizontal_flips and np.random.randint(0, 2) == 0:
# 			img = cv2.flip(img, 1)
# 			for bbox in img_data_aug['bboxes']:
# 				x1 = bbox['x1']
# 				x2 = bbox['x2']
# 				bbox['x2'] = cols - x1
# 				bbox['x1'] = cols - x2

# 		if C.use_vertical_flips and np.random.randint(0, 2) == 0:
# 			img = cv2.flip(img, 0)
# 			for bbox in img_data_aug['bboxes']:
# 				y1 = bbox['y1']
# 				y2 = bbox['y2']
# 				bbox['y2'] = rows - y1
# 				bbox['y1'] = rows - y2

# 		if C.rot_90:
# 			angle = np.random.choice([0,90,180,270],1)[0]
# 			if angle == 270:
# 				img = np.transpose(img, (1,0,2))
# 				img = cv2.flip(img, 0)
# 			elif angle == 180:
# 				img = cv2.flip(img, -1)
# 			elif angle == 90:
# 				img = np.transpose(img, (1,0,2))
# 				img = cv2.flip(img, 1)
# 			elif angle == 0:
# 				pass

# 			for bbox in img_data_aug['bboxes']:
# 				x1 = bbox['x1']
# 				x2 = bbox['x2']
# 				y1 = bbox['y1']
# 				y2 = bbox['y2']
# 				if angle == 270:
# 					bbox['x1'] = y1
# 					bbox['x2'] = y2
# 					bbox['y1'] = cols - x2
# 					bbox['y2'] = cols - x1
# 				elif angle == 180:
# 					bbox['x2'] = cols - x1
# 					bbox['x1'] = cols - x2
# 					bbox['y2'] = rows - y1
# 					bbox['y1'] = rows - y2
# 				elif angle == 90:
# 					bbox['x1'] = rows - y2
# 					bbox['x2'] = rows - y1
# 					bbox['y1'] = x1
# 					bbox['y2'] = x2        
# 				elif angle == 0:
# 					pass

# 	img_data_aug['width'] = img.shape[1]
# 	img_data_aug['height'] = img.shape[0]
# 	return img_data_aug, img
