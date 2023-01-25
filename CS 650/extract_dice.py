import cv2
from cube_detection import CubeFinder

dice_list = []


def tiny_img(image, x, y):
    r = 40
    min_y = 0 if y-r < 0 else y-r
    max_y = image.shape[0] if y+r > image.shape[0] else y+r
    min_x = 0 if x-r < 0 else x-r
    max_x = image.shape[1] if x+r > image.shape[1] else x+r
    return image[min_y: max_y, min_x: max_x]


def print_loc(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        dice_list.append([x, y])


def find_sub_images(img_location):
    image = cv2.imread(img_location)

    cube_finder = CubeFinder(["images/board/back1.jpg", "images/board/back2.jpg", "images/board/back3.jpg"])
    centers = cube_finder.find_centers(img_location)
    # print(centers)

    sub_images = []
    for pair_ in centers:
        sub_images.append(tiny_img(image, pair_[1], pair_[0]))
    return sub_images, centers


if __name__ == '__main__':
    MANUALLY = True

    if MANUALLY:
        img = cv2.imread("images/board/columns1.jpg")
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", print_loc)
        cv2.imshow("image", img)
        cv2.waitKey(0)

        # print(dice_list)

        ctr = 1
        for pair in dice_list:
            sub_img = tiny_img(img, pair[0], pair[1])
            cv2.imwrite("images/dice/all_types/unsorted/" + str(ctr) + ".jpg", sub_img)
            ctr += 1

        # sub_img = tiny_img(img, 1280, 210)
        # cv2.imshow('image', sub_img)
        # cv2.waitKey(0)
    else:
        img_loc = 'images/board/board1.jpg'

        sub_imgs, centers_ = find_sub_images(img_loc)

        ctr = 0
        for sub_img in sub_imgs:
            cv2.imwrite("images/dice/binary/others/" + str(ctr) + ".jpg", sub_img)
            ctr += 1

        # img_loc = 'images/board/board1.jpg'
        # img = cv2.imread(img_loc)
        #
        # cube_finder = CubeFinder(["images/board/back1.jpg", "images/board/back2.jpg", "images/board/back3.jpg"])
        # centers = cube_finder.find_centers(img_loc)
        # print(centers)
        #
        # ctr = 0
        # for pair in centers:
        #     sub_img = tiny_img(img, pair[big_red], pair[0])
        #     cv2.imwrite("images/dice/binary/others/" + str(ctr) + ".jpg", sub_img)
        #     ctr += big_red
