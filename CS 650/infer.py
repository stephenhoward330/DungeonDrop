from torchvision import *
import torch.nn as nn
from extract_dice import find_sub_images
import cv2
from PIL import Image


def t(c):  # flip and adjust centers to offset text location
    return c[1]-10, c[0]+10


# returns the list of centers that are columns
def infer(img_loc, show=False):
    my_transforms = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sub_images, centers = find_sub_images(img_loc)

    # convert the cv images to PIL images
    new_sub = []
    for sub_image in sub_images:
        new_sub.append(Image.fromarray(sub_image).convert('RGB'))

    tensors = [my_transforms(img) for img in new_sub]
    tensors = torch.stack(tensors)

    net = models.resnet18(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)
    net.load_state_dict(torch.load('resnet.pt'))
    net.eval()

    outputs = net(tensors)
    results = torch.argmax(outputs.data, dim=1)
    print(results)

    if show:
        image = cv2.imread(img_loc)
        for i in range(len(centers)):
            if results[i] == 1:  # NOT A COLUMN
                cv2.putText(image, "O", t(centers[i]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
            else:  # COLUMN
                cv2.putText(image, "X", t(centers[i]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    col_list = []
    for i in range(len(centers)):
        if results[i] == 0:  # A COLUMN
            col_list.append(centers[i])

    return col_list


type_dict = {'big_red': 0, 'clear': 1, 'clear_pink': 2, 'clear_blue': 3, 'column': 4, 'dice': 5, 'gold': 6,
             'green': 7, 'ice': 8, 'orange': 9, 'yellow': 10}
type_dict = {v: k for k, v in type_dict.items()}


# returns the list of centers that are columns
def infer_all(img_loc, show=False):
    my_transforms = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sub_images, centers = find_sub_images(img_loc)

    # convert the cv images to PIL images
    new_sub = []
    for sub_image in sub_images:
        new_sub.append(Image.fromarray(sub_image).convert('RGB'))

    tensors = [my_transforms(img) for img in new_sub]
    tensors = torch.stack(tensors)

    net = models.resnet18(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 11)
    net.load_state_dict(torch.load('all_cubes.pt'))
    net.eval()

    outputs = net(tensors)
    results = torch.argmax(outputs.data, dim=1)
    print(results)

    if show:
        image = cv2.imread(img_loc)
        for i in range(len(centers)):
            cv2.putText(image, type_dict[int(results[i])], t(centers[i]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    type_list = []
    for i in range(len(results)):
        type_list.append(type_dict[int(results[i])])

    return centers, type_list


if __name__ == '__main__':
    loc = 'images/board/game6.jpg'
    print(infer(loc, show=True))
    print(infer_all(loc, show=True))
