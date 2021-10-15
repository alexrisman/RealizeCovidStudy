import pydicom
import os
import subprocess
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


def load_jpg(filepath, model_name='covidnet'):
    covidnet_ind = 'covidnet' in model_name 
    img = cv2.imread(filepath)
    if covidnet_ind:
        size = 480
        img = crop_top(img, percent=.08)
        img = central_crop(img)
    else:
        size = 224
    img = cv2.resize(img, (size, size))
    img = img.astype('float32') / 255.0
    if not covidnet_ind:
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        img = (img - imagenet_mean) / imagenet_std
    return img


def dcm_to_jpg(accession_number, dcm_path):
    # https://github.com/pydicom/pydicom/issues/352
    dcm = pydicom.read_file(dcm_path)
    pixel_array = dcm.pixel_array
    jpeg_path = "{0}.jpeg".format(accession_number)
    shape = pixel_array.shape
    # Convert to float to avoid overflow or underflow losses.
    image_2d = pixel_array.astype(float)
    photometric_interpretation = dcm.PhotometricInterpretation
    if photometric_interpretation == 'MONOCHROME1':
        image_2d = np.max(image_2d) - image_2d
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    cv2.imwrite(jpeg_path, image_2d_scaled)
    return jpeg_path


def get_dicom_fields(dcm_path):
    dcm_dict = {}
    dcm = pydicom.read_file(dcm_path)
    dcm_dict['manufacturer'] = dcm.get('Manufacturer')
    return dcm_dict

