import cv2

# image resize if the size of width or height is larger than 1024
def img_resize(input_data):
    img_h, img_w, _ = input_data.shape
    if img_w > 1024:
        ratio=1024/img_w
        input_data=cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    elif img_h > 1024:
        ratio = 1024/img_h
        input_data=cv2.resize(input_data, (int(ratio*img_w), int(ratio*img_h)))
    return input_data