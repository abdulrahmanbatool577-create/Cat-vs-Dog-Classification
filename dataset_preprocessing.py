
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def extract_color_features(image):
    """Extract BGR mean/std and HSV histogram features."""
    bgr_mean = np.mean(image, axis=(0, 1))
    bgr_std = np.std(image, axis=(0, 1))
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
    return np.hstack([bgr_mean, bgr_std, hsv_hist])

def extract_hog_features(image, img_size=(64, 64)):
    """Extract Histogram of Oriented Gradients (HOG) descriptors."""
    gray = cv2.cvtColor(cv2.resize(image, img_size), cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

def process_single_image(image_path):
    """Preprocess image and extract hybrid color + shape feature vector."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    color_feats = extract_color_features(image)
    hog_feats = extract_hog_features(image)
    return np.hstack([color_feats, hog_feats])
