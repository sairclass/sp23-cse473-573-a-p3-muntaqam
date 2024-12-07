'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''

import torch

import face_recognition

from typing import Dict, List


from utils import show_image
import utils
'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''



def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []  # Please make sure your output follows this data format.

    # TODO: Add your code here. Do not modify the return and input arguments.
    img_temp = utils.bgr_to_rgb(img)
    face_locations = face_recognition.face_locations(img_temp.permute(1, 2, 0).numpy(), number_of_times_to_upsample=2,
                                                     model="hog")

    for face_location in face_locations:
        detection_results.append(convert_and_trim_bb(img.shape[-2:], face_location))

    # show image with bounding box
    # for face_location in face_locations:
    #     top, right, bottom, left = face_location
    #     # Draw a box around the face
    #     cv2.rectangle(img_temp.permute(1, 2, 0).numpy(), (left, top), (right, bottom), (0, 0, 255), 2)
    # show_image(img_temp)
    # print(detection_results)
    return detection_results


def convert_and_trim_bb(image_shape, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box ## convert_and_trim_bb([418,450], face_location)
    startX = rect[3]
    startY = rect[0]
    endX = rect[1]
    endY = rect[2]
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image_shape[1])
    endY = min(endY, image_shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return [float(startX), float(startY), float(w), float(h)]


# import utils
# img = utils.read_image("validation_folder/images/img_1.jpg")
# print(detect_faces(img))


def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[]] * K  # Please make sure your output follows this data format.

    # TODO: Add your code here. Do not modify the return and input arguments.
    imgs_embds = {}
    for img_name in imgs:
        img = imgs[img_name]
        img_temp = utils.bgr_to_rgb(img)
        face_locations = face_recognition.face_locations(img_temp.permute(1, 2, 0).numpy(),
                                                         number_of_times_to_upsample=1,
                                                         model="hog")
        if len(face_locations) == 0:
            continue
        else:
            face_encodings = face_recognition.face_encodings(img_temp.permute(1, 2, 0).numpy(), face_locations)
            face_encoding = face_encodings[0]
            imgs_embds[img_name] = face_encoding

    clusters = kmeans(imgs_embds, K)

    cluster_results = list(clusters.values())
    print("Task 2 results: ", cluster_results)
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''


# TODO: Your functions. (if needed)
def kmeans(imgs: Dict[str, torch.Tensor], K: int) -> Dict[int, list]:
    # convert the image data to a tensor
    img_data = torch.Tensor(list(imgs.values()))
    # print(list(imgs.keys()))

    img_data = img_data.reshape([len(img_data), -1])
    # randomly initialize K centroids
    rp = torch.randperm(len(imgs))[:K]
    centroids = img_data[rp, :]

    # repeat until convergence
    while True:
        # assign each data point to its closest centroid
        distances = torch.cdist(img_data, centroids)
        closest_centroids = torch.argmin(distances, dim=1)
        # print(closest_centroids)
        # update each centroid to be the mean of its assigned data points
        for i in range(K):
            centroids[i] = img_data[closest_centroids == i].mean(dim=0).reshape([1,-1])

        # check for convergence
        if torch.all(torch.eq(closest_centroids, torch.argmin(torch.cdist(img_data, centroids), dim=1))):
            break

    # group the images by their assigned centroid
    groups = {i: [] for i in range(K)}
    for i, name in enumerate(imgs.keys()):
        groups[closest_centroids[i].item()].append(name)

    return groups
