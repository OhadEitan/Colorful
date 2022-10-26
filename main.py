import streamlit as st
import cv2 as cv
from random import randint
import os
import numpy as np
from copy import copy, deepcopy
from PIL import Image


# building the most popular Colors  in HSV format
class ColorShades:
    def __init__(self, lower_h, upper_h, lower_s, upper_s, lower_v, upper_v, name):
        self.lower_h = lower_h
        self.upper_h = upper_h
        self.lower_s = lower_s
        self.upper_s = upper_s
        self.lower_v = lower_v
        self.upper_v = upper_v
        self.name = name


red = ColorShades(165, 10, 0, 255, 22, 254, "red")
orange = ColorShades(8, 16, 0, 200, 200, 254, "orange")
yellow = ColorShades(17, 27, 0, 255, 22, 254, "yellow")
light_green = ColorShades(28, 49, 0, 255, 22, 254, "light_green")
green = ColorShades(50, 69, 0, 255, 22, 254, "green")
light_blue = ColorShades(70, 90, 0, 255, 22, 254, "light_blue")
blue = ColorShades(90, 115, 0, 255, 22, 254, "blue")
purple = ColorShades(116, 140, 0, 255, 22, 254, "purple")
pink = ColorShades(140, 170, 0, 255, 22, 254, "pink")
black = ColorShades(0, 360, 0, 255, 0, 22, "black")
white = ColorShades(0, 360, 0, 255, 255, 255, "white")
brown = ColorShades(13, 16, 150, 200, 160, 200, "brown")

all_color_shades = [red, orange, yellow, light_green, green, light_blue, blue, purple, pink
    , black, white, brown]


class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages.append({

            "title": title,
            "function": func
        })

    def run(self):
        # Drodown to select the page to run
        page = st.sidebar.selectbox(
            'App Navigation',
            self.pages,
            format_func=lambda page: page['title']
        )

        # # run the app function
        # page['function']()
        #


# showing the HSV param of specific pixel
def mouseHSV(event, x, y, ):
    if event == cv.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colorsB = img[y, x, 0]
        colorsG = img[y, x, 1]
        colorsR = img[y, x, 2]
        colors = img[y, x]
        hsv_value = np.uint8([[[colorsB, colorsG, colorsR]]])
        hsv = cv.cvtColor(hsv_value, cv.COLOR_BGR2HSV)
        print("HSV : ", hsv)
        # print(" of pixel: X: ",x,"Y: ",y)


def rescaleFrame(frame, scale=3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


def list_of_color_in_img(p_h, p_s, p_v):
    out_list = []
    height, weight = p_h.shape[:2]
    for x in range(height):
        for y in range(weight):
            for shade in all_color_shades:
                if shade.lower_h <= p_h[x, y] <= shade.upper_h and \
                        shade.lower_v <= p_v[x, y] <= shade.upper_v and shade not in out_list:
                    if shade.name != "white" and shade.name != "black":
                        out_list.append(shade)
                        break
    return out_list


def color_head2head_calc(h_pixel_in, old_color, new_color):
    scale_from = (h_pixel_in - old_color.lower_h) / (old_color.upper_h - old_color.lower_h)
    h_pixel_out = (new_color.upper_h - new_color.lower_h) * scale_from + new_color.lower_h
    return h_pixel_out


def v_color_brightness(v_pixel_in, percent):
    v_pixel_out = v_pixel_in * (1 + (percent / 100))
    if v_pixel_out > 255:
        v_pixel_out = 255
    return v_pixel_out


def head2head_filter(p_h, old_color, new_color):
    height, weight = p_h.shape[:2]
    new_h = deepcopy(p_h)
    # format_hsv = cv.merge([p_h, p_s, p_v])
    # format_rgb = cv.cvtColor(format_hsv, cv.COLOR_HSV2BGR)
    for x in range(height):
        for y in range(weight):
            if old_color.lower_h <= p_h[x, y] <= old_color.upper_h:
                new_h[x, y] = color_head2head_calc(p_h[x, y], old_color, new_color)
    return new_h


def brightness_filter(p_h, p_v, color, brightness_percent):
    height, weight = p_h.shape[:2]
    new_v = deepcopy(p_v)
    for x in range(height):
        for y in range(weight):
            if color.lower_h <= p_h[x, y] <= color.upper_h:
                new_v[x, y] = v_color_brightness(p_v[x, y], brightness_percent)
    return new_v


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
# img = cv.imread('q_test/test_5.webp', cv.IMREAD_UNCHANGED)
# # img = cv.imread('birth_of_universe.jpeg', cv.IMREAD_UNCHANGED)
# img = rescaleFrame(img)

# convert to HSV
# hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# h, s, v = cv.split(hsv)

# cv.namedWindow('mouseHSV')
# cv.setMouseCallback('mouseHSV',mouseHSV)
# while(1):
#     cv.imshow('mouseHSV',img)
#     if cv.waitKey(20) & 0xFF == 27:
#         break
# #if esc pressed, finish.
# cv.destroyAllWindows()


# out_list_after_func = list_of_color_in_img(h, s, v)
# print(out_list_after_func)
# print(len(all_color_shades) - len(out_list_after_func))
# h, s, v = head2head_filter(h, s, v,blue , yellow)
# h, s, v = brighness_filter(h, s, v,yellow , -70)
# cv.imshow('filterd', filtered_to_display(h, s, v))
# cv.imshow('source', img)
#
# cv.waitKey(0)
# cv.destroyAllWindows()


# if a person mistake a picture show him some filtered images and let him choose his fave
# asking user to tell me if he see all colrs as he should
# num_of_correct_answers = 0
# num_of_filter = 0
# mistake = False
# while num_of_correct_answers < 5:
#
#     if mistake:
#         num_of_correct_answers = 0
#         num_of_filter = num_of_filter + 1


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


# def brighten_image(image, amount):
#     img_bright = cv.convertScaleAbs(image, beta=amount)
#     return img_bright
#
#
# def blur_image(image, amount):
#     blur_img = cv.GaussianBlur(image, (11, 11), amount)
#     return blur_img
#
#
# def enhance_details(img):
#     hdr = cv.detailEnhance(img, sigma_s=12, sigma_r=0.15)
#     return hdr

def page_1():
    st.title("Colorful Demo App")
    st.subheader("This app allows color-blind people use their electronic devices as no color-blind people")
    st.text("We use OpenCV and Streamlit for this demo")
    st.subheader("Our main goal is to find your customise color filter")


def proc():
    st.write(st.session_state.text_key)


def page_2():
    # "st.session_state object:", st.session_state
    image, image_name, str_value = random_pic()
    st.image(image, "file name: " + image_name, 200)
    correct_answer = 0
    answer_file_part_name = "solution_" + str_value
    for file in os.listdir("./s_test"):
        if answer_file_part_name in file:
            full_correct_file_name = "./s_test/" + file
            with open(full_correct_file_name, 'r') as f:
                correct_answer = f.read()
    st.text_input("enter the number you see here", key="text1")
    # slider_answer = st.slider("enter the number you see here", min_value=1, max_value=100, step=1)
    st.write(st.session_state.text1)
    if st.button("submit your answer"):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        # colors = list_of_color_in_img(h, s, v)
        if str(st.session_state.text1) == correct_answer:
            st.text("Good job")
        else:
            st.text("The answer was incorrect pls choose the picture where you can clearly\nsee the number: " +
                    correct_answer)
            colors = list_of_color_in_img(h, s, v)
            st.write(len(colors))
            if "first_test_color_1" not in st.session_state:
                st.session_state["first_test_color_1"] = colors[randint(0, len(colors) - 1)]
            if "first_test_color_2" not in st.session_state:
                st.session_state["first_test_color_2"] = colors[randint(0, len(colors) - 1)]
            if "second_test_color_1" not in st.session_state:
                st.session_state["second_test_color_1"] = colors[randint(0, len(colors) - 1)]
            if "second_test_color_2" not in st.session_state:
                st.session_state["second_test_color_2"] = colors[randint(0, len(colors) - 1)]
            if "third_test_color_1" not in st.session_state:
                st.session_state["third_test_color_1"] = colors[randint(0, len(colors) - 1)]

            # st.write("first h&h filter: color 1 is - " + st.session_state.first_test_color_1.name +
            #          " color 2 is - " + st.session_state.first_test_color_2.name)
            # st.write("second h&h filter: color 1 is - " + st.session_state.second_test_color_1.name +
            #          " color 2 is - " + st.session_state.second_test_color_2.name)
            # st.write("first b filter: color 1 is - " + st.session_state.third_test_color_1.name)

            # first filter
            h_1 = head2head_filter(h, st.session_state.first_test_color_1,
                                   st.session_state.first_test_color_2)
            filtered_hsv_1 = cv.merge([h_1, s, v])
            filtered_bgr_1 = cv.cvtColor(filtered_hsv_1, cv.COLOR_HSV2BGR)
            st.image(filtered_bgr_1, "test_after_filter ", 200)

            # second filter
            h_2 = head2head_filter(h, st.session_state.second_test_color_1,
                                   st.session_state.second_test_color_2)
            filtered_hsv_2 = cv.merge([h_2, s, v])
            filtered_bgr_2 = cv.cvtColor(filtered_hsv_2, cv.COLOR_HSV2BGR)
            st.image(filtered_bgr_2, "test_after_filter ", 200)

            # third filter
            v_3 = brightness_filter(h, v, st.session_state.third_test_color_1, -70)
            filtered_hsv_3 = cv.merge([h, s, v_3])
            filtered_bgr_3 = cv.cvtColor(filtered_hsv_3, cv.COLOR_HSV2BGR)
            st.image(filtered_bgr_3, "test_after_filter ", 200)




def page_3():
    st.text_input("enter the number of filtered pictures where you see the number the best", key="text2")
    st.write(st.session_state.text2)
    st.title("Upload a picture")
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    # file = st.file_uploader("Upload image file", accept_multiple_files=False)
    # if file is not None:
    #     image_bytes = file.getvalue()
    st.write("Here's what you uploaded!")
    st.image(original_image, width=400)
    # image = cv.imread(original_image, cv.IMREAD_UNCHANGED)
    hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    if st.session_state.text2 == '1':
        h_1 = head2head_filter(h, st.session_state.first_test_color_1,
                               st.session_state.first_test_color_2)
        filtered_hsv_1 = cv.merge([h_1, s, v])
        filtered_bgr_1 = cv.cvtColor(filtered_hsv_1, cv.COLOR_HSV2BGR)
        st.image(filtered_bgr_1, "uploaded picture after filter ", width=400)

    if st.session_state.text2 == '2':
        h_2 = head2head_filter(h, st.session_state.second_test_color_1,
                               st.session_state.second_test_color_2)
        filtered_hsv_2 = cv.merge([h_2, s, v])
        filtered_bgr_2 = cv.cvtColor(filtered_hsv_2, cv.COLOR_HSV2BGR)
        st.image(filtered_bgr_2, "uploaded picture after filter ", width=400)

    if st.session_state.text2 == '3':
        v_3 = brightness_filter(h, v, st.session_state.third_test_color_1, -70)
        filtered_hsv_3 = cv.merge([h, s, v_3])
        filtered_bgr_3 = cv.cvtColor(filtered_hsv_3, cv.COLOR_HSV2BGR)
        st.image(filtered_bgr_3, "uploaded picture after filter ", width=400)

    # uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     bytes_data = uploaded_file.read()
    #     st.write("filename:", uploaded_file.name)
    #     img = cv.imread(uploaded_file.name, cv.IMREAD_UNCHANGED)
    #     st.image(img, "file name: ", 250)


# def page_2_after_first(correct_answer, new_correct_file_name):
#     image = Image.open(   new_correct_file_name)
#     st.image(image, "file name: " + new_correct_file_name, 400)
#     if st.session_state.text == correct_answer:
#         st.text("Good job")
#     else:
#         st.text("the correct answer was: " + correct_answer)


# def random_number():
#     if st.session_state.random == -99:
#         value = randint(1, 15)
#         st.session_state.random = value
#         st.text("in random_number: " + str(st.session_state.random))
#     return st.session_state.random


def random_pic():
    # st.session_state.random = random_number()
    str_random = str(st.session_state.rn)
    file_name = "test_" + str_random
    correct_file_name = ""
    for file in os.listdir("./q_test"):
        if file_name in file:
            correct_file_name = file
    new_correct_file_name = "./q_test/" + correct_file_name
    # image = Image.open(new_correct_file_name)
    image = cv.imread(new_correct_file_name, cv.IMREAD_UNCHANGED)
    st.text("in random_pic: " + str(st.session_state.rn))

    return image, new_correct_file_name, str_random


def change_number():
    st.session_state["rn"] = randint(1, 15)
    return


def main_loop():
    st.set_page_config(page_title="Colorful Demo App", page_icon="👋")
    st.session_state.page_select = st.sidebar.radio('Pages', ['Page 1', 'Page 2', 'Page 3'])

    st.session_state.page_select == 'Page 1'
    if st.session_state.page_select == 'Page 1':
        page_1()

    if st.session_state.page_select == 'Page 2':
        if "rn" not in st.session_state:
            st.session_state["rn"] = randint(1, 15)
        st.session_state["test_num_1"] = -987
        page_2()

    if st.session_state.page_select == 'Page 3':
        page_3()

    # blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    # brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    # apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    # Image.open(image_file)
    # original_image = np.array(original_image)
    #
    # processed_image = blur_image(original_image, blur_rate)
    # processed_image = brighten_image(processed_image, brightness_amount)
    #
    # if apply_enhancement_filter:
    #     processed_image = enhance_details(processed_image)
    #
    # st.text("Original Image vs Processed Image")
    # st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()
