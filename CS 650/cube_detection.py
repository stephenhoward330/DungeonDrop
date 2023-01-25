import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import feature
import scipy.linalg as la
from scipy.stats import skew
import math
from functools import reduce
import cv2
import numpy
from scipy import ndimage
from sklearn.cluster import KMeans


class CubeFinder:

    def __init__(self, backgrounds, tau=120, k=51, stride=10, min_score=15, min_count=250, r=25,
                 alpha=1, max_width=50, ratio=.4, num_points=100, min_dist=75):

        BACKS = np.asarray([np.asarray(Image.open(i)) for i in backgrounds])
        self.beta = np.mean(BACKS, axis=0)
        self.tau = tau
        self.k = k
        self.stride = stride
        self.min_score = min_score
        self.min_count = min_count
        self.r = r
        self.alpha = alpha
        self.max_width = max_width
        self.ratio = ratio
        self.num_points = num_points
        self.min_dist = min_dist

    def find_centers(self, img_path):

        # Load image
        self.img = np.asarray(Image.open(img_path))

        # Preprocess to get preprocessed image
        self.preprocess(self.img, self.beta, self.tau)

        # Then run feature detection
        self.feature_detection(self.img_prime, self.k, self.stride, self.min_score, self.min_count)

        # After feature detection, run voting method
        self.voting_method(self.img_prime, self.feature_points_y, self.feature_points_x, self.r, self.alpha)

        # After voting method, get bounding boxes
        self.bounding_boxes(self.votes, self.feature_points_x, self.feature_points_y, self.max_width)

        # From the bounding boxes, get your final cube center points
        return self.identify_points(self.boxes, self.feature_points_x, self.feature_points_y, self.img,
                                    self.ratio, self.num_points, self.min_dist)

    def preprocess(self, img, background, tau):
        """ Takes in an image, and performs preprocessing. """

        # Do background detection
        diff = np.abs(la.norm(img - background, ord=2, axis=2))
        # mask = diff > tau

        # Now do edge detection
        edgey = ndimage.sobel(diff, 1)  # vertical derivative
        edgex = ndimage.sobel(diff, 0)  # horizontal derivative
        edge = np.sqrt(edgey ** 2 + edgex ** 2)

        borders = np.where(edge > tau, 1, 0)
        self.img_prime = borders

    def feature_detection(self, img, k, stride, min_score, min_count):
        """ Uses skew estimate as local invariant feature of cubes. """

        # Make sure size is odd
        assert k % 2 == 1

        min_x = []
        min_y = []

        w = int(np.floor(k / 2))

        for i in range(w, img.shape[0] - w, stride):
            for j in range(w, img.shape[1] - w, stride):
                M = img[i - w:i + w + 1, j - w:j + w + 1]

                if np.sum(M) > min_count:
                    dist = []
                    for x in range(k):
                        for y in range(k):
                            if M[x, y]:
                                dist.append((np.sqrt((x - w) ** 2 + (y - w) ** 2)))

                    symmetry = np.abs(np.mean(dist) - np.percentile(dist, .5))

                    if symmetry < min_score:
                        min_x.append(j)
                        min_y.append(i)

        self.feature_points_x = np.array(min_y)
        self.feature_points_y = np.array(min_x)

        # return np.array(min_x), np.array(min_y)

    def voting_method(self, img, x_cords, y_cords, r, alpha):
        """ Uses voting to determine likely points of cube. """

        L = np.zeros(img.shape)

        for x, y in zip(y_cords, x_cords):  # Switched because of how matplotlib displays images...
            window_x = np.arange(max(x - r, 0), min(x + r + 1, L.shape[0]), 1)
            window_y = np.arange(max(y - r, 0), min(y + r + 1, L.shape[1]), 1)

            for wx in window_x:
                for wy in window_y:
                    if (wx - x) ** 2 + (wy - y) ** 2 <= r ** 2:
                        L[wx, wy] += 1  # Uniform weighting
                        # L[wx,wy] += (alpha/r**dice)*(r**dice-np.sqrt((wx-x)**dice+(wy-y)**dice)) # Linear distance
                        # L[wx,wy] = big_red # Single vote
        self.votes = L

    def bounding_boxes(self, L, features_x, features_y, max_width):
        """ Takes votes, and using a greedy algorithm, finds bounding boxes. """

        centers = [(x, y, L[x, y]) for x, y in zip(features_x, features_y)]

        centers.sort(key=lambda point: point[2])
        centers.reverse()

        boxes = []

        def find_bounding_box(x, y):
            """ Given an x,y, find the smallest bounding box around our voting point. """

            n, w, s, e = 2, 1, 1, 2

            i = 0
            while True:

                # Get window
                window = L[max(x - (w + i), 0):min(x + (e + i), L.shape[0]),
                         max(y - (s + i), 0):min(L.shape[1], y + (n + i))]

                # Get edge strengths
                edge_u = sum(window[0, :])
                edge_d = sum(window[-1, :])
                edge_l = sum(window[:, 0])
                edge_r = sum(window[:, -1])
                total_strength = edge_l + edge_r + edge_u + edge_d

                if total_strength == 0 or i > max_width:
                    return (x, y, i)
                else:
                    i += 1

        used_points = []
        for point in list(centers):
            if point not in used_points:
                res = find_bounding_box(point[0], point[1])
                x_bounds = (max(point[0] - res[2], 0), min(point[0] + res[2], L.shape[0]))
                y_bounds = (max(point[1] - res[2], 0), min(point[1] + res[2], L.shape[1]))
                for p in centers:
                    if (p[0] > x_bounds[0]) and (p[0] < x_bounds[1]) and (p[1] > y_bounds[0]) and (p[1] < y_bounds[1]):
                        used_points.append(p)
                boxes.append((res))
            else:
                pass

        self.boxes = boxes

    def plot_bounding_boxes(self):
        """ Plots the bounding boxes over the images. """

        plt.figure(figsize=(8, 8))

        for b in self.boxes:
            x_bounds = (max(b[0] - b[2], 0), min(b[0] + b[2], self.img.shape[0]))
            y_bounds = (max(b[1] - b[2], 0), min(b[1] + b[2], self.img.shape[1]))

            plt.imshow(self.img)
            plt.vlines(y_bounds[0], x_bounds[0], x_bounds[1], color='red')
            plt.vlines(y_bounds[1], x_bounds[0], x_bounds[1], color='red')
            plt.hlines(x_bounds[0], y_bounds[0], y_bounds[1], color='red')
            plt.hlines(x_bounds[1], y_bounds[0], y_bounds[1], color='red')
        plt.show()

    def identify_points(self, boxes, x_points, y_points, img, ratio, num_points, min_dist):
        """ Takes in bounding boxes, finds correlation, and places big_red or dice cubes as needed. """
        final_points = []

        for b in boxes[:]:
            x_bounds = (max(b[0] - b[2], 0), min(b[0] + b[2], img.shape[0]))
            y_bounds = (max(b[1] - b[2], 0), min(b[1] + b[2], img.shape[1]))

            window = []
            for x, y in zip(x_points, y_points):
                if (x > x_bounds[0]) and (x < x_bounds[1]) and (y > y_bounds[0]) and (y < y_bounds[1]):
                    window.append((x, y))

            # Get (x,y) distribution of features
            W = np.zeros((2, len(window)))
            W[0, :] = [w_[0] for w_ in window]
            W[1, :] = [w_[1] for w_ in window]

            # If we only have one point, add it
            if W.shape[1] == 1:
                final_points.append((b[0], b[1]))

            else:
                # Compute covariance
                sigma = np.cov(W)
                a = max(la.eigvals(sigma))
                b = min(la.eigvals(sigma))

                # This means we have two clusters
                if np.abs((b / a)) < ratio or W.shape[1] >= num_points:

                    # Use K-Means to find dice clusters
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(W.T)
                    centers = kmeans.cluster_centers_
                    point1 = centers[0, :]
                    point2 = centers[1, :]

                    # If the centers are closer than 50 in L2 norm, merge to one again
                    if la.norm(point1 - point2) < min_dist:
                        final_points.append((int(np.mean(W[0, :])), int(np.mean(W[1, :]))))

                    else:
                        final_points.append((int(point1[0]), int(point1[1])))
                        final_points.append((int(point2[0]), int(point2[1])))

                # This means we have one cube
                else:
                    final_points.append((int(np.mean(W[0, :])), int(np.mean(W[1, :]))))

        return final_points
