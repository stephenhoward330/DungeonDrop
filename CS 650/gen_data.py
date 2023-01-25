import cv2
import os

# saves a 90-, 180-, and 270- rotation of each image
if __name__ == '__main__':
    for drt in os.listdir("images/dice/all_types"):
        path = os.path.join("images/dice/all_types", drt)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path)
            img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_180 = cv2.rotate(img, cv2.ROTATE_180)
            img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(file_path[:-4] + "_90.jpg", img_90)
            cv2.imwrite(file_path[:-4] + "_180.jpg", img_180)
            cv2.imwrite(file_path[:-4] + "_270.jpg", img_270)
