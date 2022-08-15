import cv2 as cv
import numpy as np
import streamlit as sl
import pandas as pd
from scipy.spatial import KDTree

sl.write('hi')

class Color_shadeds:
    def __init__(self, lower_h, upper_h, lower_s, upper_s,lower_v,upper_v):
        self.lower_h = lower_h
        self.upper_h = upper_h
        self.lower_s = lower_s
        self.upper_s = upper_s
        self.lower_v = lower_v
        self.upper_v = upper_v



red = Color_shadeds(165, 10, 0, 255, 22, 255)
orange = Color_shadeds(10, 25, 0, 255, 22, 255)
yellow = Color_shadeds(25, 45, 0, 255, 22, 255)
light_green = Color_shadeds(45, 60, 0, 255, 22, 255)
green = Color_shadeds(60, 80, 0, 255, 22, 255)
light_blue = Color_shadeds(80, 100, 0, 255, 22, 255)
blue = Color_shadeds(100, 125, 0, 255, 22, 255)
purple = Color_shadeds(125, 140, 0, 255, 22, 255)
pink = Color_shadeds(140, 165, 0, 255, 22, 255)
black = Color_shadeds(0, 360, 0, 255, 0, 22)
white = Color_shadeds(0, 360, 0, 5, 245, 255)
brown = Color_shadeds(13, 16, 150, 200, 160, 190)



def mouseRGB(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colorsB = img[y, x, 0]
        colorsG = img[y, x, 1]
        colorsR = img[y, x, 2]
        colors = img[y, x]
        hsv_value = np.uint8([[[colorsB, colorsG, colorsR]]])
        hsv = cv.cvtColor(hsv_value, cv.COLOR_BGR2HSV)
        print("HSV : ", hsv)
        # print(" of pixel: X: ",x,"Y: ",y)


def rescaleFrame(frame, scale=0.15):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


def color_head2head_calc(h_pixel_in, old_color, new_color):
    scale_from = (h_pixel_in - old_color.lower_h) / (old_color.upper_h - old_color.lower_h)
    h_pixel_out = (new_color.upper_h - new_color.lower_h) * scale_from + new_color.lower_h
    return h_pixel_out


def v_color_brightness(v_pixel_in, precent):
    v_pixel_out = v_pixel_in * (1 + (precent / 100))
    if (v_pixel_out > 255):
        v_pixel_out = 255
    return v_pixel_out


# def s_color_brightness(s_color_in, precent):
#     s_out_color = s_color_in / (1+ (precent/100))
#     if(s_out_color <15):
#         s_out_color +=10
#     return s_out_color

def head2head_filter(p_h, p_s, p_v, old_color, new_color):
    height, weight = h.shape[:2]
    # format_hsv = cv.merge([p_h, p_s, p_v])
    # format_rgb = cv.cvtColor(format_hsv, cv.COLOR_HSV2BGR)
    for x in range(height):
        for y in range(weight):
            if (p_h[x, y] >= old_color.lower_h and p_h[x, y] <= old_color.upper_h):
                p_h[x, y] = color_head2head_calc(p_h[x, y], old_color, new_color)
    return p_h, p_s, p_v


def brighness_filter(p_h, p_s, p_v, color, brightness_precent):
    height, weight = h.shape[:2]
    for x in range(height):
        for y in range(weight):
            if (p_h[x, y] >= color.lower_h and p_h[x, y] <= color.upper_h):
                p_v[x, y] = v_color_brightness(p_v[x, y], brightness_precent)
    return p_h, p_s, p_v


def filtered_to_display(p_h, p_s, p_v):
    height, weight = p_h.shape[:2]
    # checking values are not out of range
    for x in range(height):
        for y in range(weight):
            if p_h[x, y] > 360:
                print('error')
            if p_v[x, y] > 255:
                p_v[x, y] = 255
            if p_v[x, y] < 0:
                p_v[x, y] = 0
            if p_s[x, y] > 255:
                p_s[x, y] = 255
            if p_s[x, y] < 0:
                p_s[x, y] = 0

    # re-building HSV and converting it into BGR
    filtered_hsv = cv.merge([p_h, p_s, p_v])
    filtered_bgr = cv.cvtColor(filtered_hsv, cv.COLOR_HSV2BGR)
    return filtered_bgr


# load image with alpha channel
img = cv.imread('birth_of_universe.jpeg', cv.IMREAD_UNCHANGED)
img = rescaleFrame(img)

# convert to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)

# cv.namedWindow('mouseRGB')
# cv.setMouseCallback('mouseRGB',mouseRGB)
# while(1):
#     cv.imshow('mouseRGB',img)
#     if cv.waitKey(20) & 0xFF == 27:
#         break
# #if esc pressed, finish.
# cv.destroyAllWindows()


h, s, v = head2head_filter(h, s, v,blue , pink)
cv.imshow('filterd', filtered_to_display(h, s, v))
cv.imshow('source', img)
cv.waitKey(0)
cv.destroyAllWindows()



#if a person mistake a picture show him some filtered images and let him choose his fave
#asking user to tell me if he see all colrs as he should
num_of_correct_answers = 0
num_of_filter = 0
mistake = False
while num_of_correct_answers < 5:

    if mistake:
        num_of_correct_answers = 0
        num_of_filter = num_of_filter + 1



# #head to head
# for x in range(height):
#     for y in range(weight):
#         if(h[x,y] >= h_green_low and h[x,y] <= h_green_high   ):
#             #if not(h[x,y] <16  and v[x,y] < 40 and s[x,y] <40):
#             h[x,y] = color_head2head(h[x,y], h_green_low,h_green_high, h_purple_low, h_purple_high)
#           #  else:
#               #  print('hellow')
#   #


# britness
# for x in range(height):
#      for y in range(weight):
#          if(h[x,y] >= h_blue_low and h[x,y] <= h_blue_high and v[x,y] <200 ):
#             v[x,y] = v_color_brightness(v[x,y] , 50)
#           #  s[x, y] = s_color_brightness(v[x, y], 40)


# cv.imshow('filterd',bgr_new)
# cv.imshow('source',img)
# cv.waitKey(0)
# cv.destroyAllWindows()


