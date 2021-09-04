# -*- coding: utf-8 -*-

import streamlit as st
from sections.model import *
from sections.introd import *

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.title("Sections")
section = st.sidebar.radio(
    "Go to:",
    ( 'Intro', 'Model'))

if section == 'Model':
    model()
elif section == 'Intro':
    introd()





