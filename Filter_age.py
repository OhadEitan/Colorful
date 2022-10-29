import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
import cv2 as cv
from random import randint
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title="Mapping Demo", page_icon="art")