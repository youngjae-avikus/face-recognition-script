#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:22:54 2023

@author: yyj
"""


# 1. Import lib
import face_recognition
from PIL import Image, ImageDraw

# 2. Load image file as a numpy array
face_image = face_recognition.load_image_file("images/testing/trump-modi.jpg")

# 3. get the face landmarks list
face_landmarks_list = face_recognition.face_landmarks(face_image)

# convert the numpy array into pil image object
pil_image = Image.fromarray(face_image)
# convert the pil image to draw object
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:    
    # join each face landmark points
    d.line(face_landmarks['chin'], fill=(255,255,255), width=2)
    d.line(face_landmarks['left_eyebrow'], fill=(255,255,255), width=2)
    d.line(face_landmarks['right_eyebrow'], fill=(255,255,255), width=2)
    d.line(face_landmarks['nose_bridge'], fill=(255,255,255), width=2)
    d.line(face_landmarks['nose_tip'], fill=(255,255,255), width=2)
    d.line(face_landmarks['left_eye'], fill=(255,255,255), width=2)
    d.line(face_landmarks['right_eye'], fill=(255,255,255), width=2)
    d.line(face_landmarks['top_lip'], fill=(255,255,255), width=2)
    d.line(face_landmarks['bottom_lip'], fill=(255,255,255), width=2)

# show the final image
pil_image.show()

# save image
pil_image.save("images/samples/abhi_landmarks.jpg")
