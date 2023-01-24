#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 12:29:58 2023

@author: yyj
"""

# Import lib
import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np

webcam_face_stream = cv2.VideoCapture(0)

while True:
    ret, face_image = webcam_face_stream.read()
    
    if not ret:
        break
    
    # small_face_image = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)
    
    # get the face landmarks list
    face_landmarks_list = face_recognition.face_landmarks(face_image)
    
    # convert the numpy array into pil image object
    pil_image = Image.fromarray(face_image)
    # convert the pil image to draw object "RGBA"
    d = ImageDraw.Draw(pil_image)
    
    for face_landmarks in face_landmarks_list:
        # Make left, right eyebrows darker 
        # Polygon on top and line on bottom with dark color
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)


        # Add lipstick to top and bottom lips
        # using red polygons and lines filled with red
        d.polygon(face_landmarks['top_lip'], fill=(0, 0, 150, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 150, 128))
        d.line(face_landmarks['top_lip'], fill=(0, 0, 150, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(0, 0, 150, 64), width=8)

        # Make left and right eyes filled with red
        d.polygon(face_landmarks['left_eye'], fill=(0, 0, 255, 100))
        d.polygon(face_landmarks['right_eye'], fill=(0, 0, 255, 100))

        # Eyeliner to left and right eyes as lines
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    
    # show the final image
    image = np.asarray(pil_image)
    cv2.imshow('face landmark', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam_face_stream.release()
cv2.destroyAllWindows()
    
    