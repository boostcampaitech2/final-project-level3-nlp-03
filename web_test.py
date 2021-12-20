import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import re

import inference as inf

st.image("logo.png")

st.write("'Between Spaces'는 Bert 기반 한국어 띄어쓰기 모델입니다. 이러이러 저러저러 텍스트 텍스트 설명 설명.")
st.write("")

text_input = st.text_input("띄어쓰기 할 텍스트를 입력해주세요. (256자 제한)", max_chars=256)

col1, col2, col3, col4, col5= st.columns(5)
col1.write("")
col2.write("")
if col3.button("띄어쓰기!"):
    if text_input:
        with st.spinner("잠시 기다려주세요..."):
            time.sleep(2)
            #result = inf(text_input, model) #model은 없어도 됨.
            result = inf.inf(text_input, 1)
        st.code(result, language="markdown")
    else:
        st.warning('텍스트를 입력해주세요.')
col4.write("")
col5.write("")
st.write("")
st.write("")
st.write("")
st.write("")

s1 = f"""<style>div.stButton > button:first-child {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px;}}<style>"""
st.markdown(s1, unsafe_allow_html=True)

s2 = f"""<style>div.stButton > button:focus:not {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px;}}<style>"""
st.markdown(s2, unsafe_allow_html=True)

s3 = f"""<style>div.stButton > button:focus: {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px;color:#50bcdf}}<style>"""
st.markdown(s3, unsafe_allow_html=True)

s4 = f"""<style>div.stButton > button:focus:not(:active) {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px; background-color:#ffffff;color:#50bcdf;box-shadow:rgba(200, 200, 200, 10) 0px 0px 0px 0.1rem;}}<style>"""
st.markdown(s4, unsafe_allow_html=True)

s5 = f"""<style>div.stButton > button:hover {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px;color:#50bcdf}}<style>"""
st.markdown(s5, unsafe_allow_html=True)

s6 = f"""<style>div.css-ns78wr:focus:not(:active) {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px;color:#ffffff}}<style>"""
st.markdown(s6, unsafe_allow_html=True)

s7 = f"""<style>div.stButton > button:active {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px;background-color:#50bcdf;color:#ffffff;box-shadow:rgba(200, 200, 200, 10) 0px 0px 0px 0.2rem;}}<style>"""
st.markdown(s7, unsafe_allow_html=True)

s8 = f"""<style>div.streamlit-expanderHeader > button:active {{ border: 1px solid #50bcdf; border-radius:10px 10px 10px 10px;color:#50bcdf;box-shadow:rgba(200, 200, 200, 10) 0px 0px 0px 0.2rem;}}<style>"""
st.markdown(s8, unsafe_allow_html=True)

with st.expander('made by'):
    git_col1, git_col2, git_col3, git_col4, git_col5, git_col6 = st.columns(6)
    git_col1.markdown('<p align="center"><a href="https://github.com/promisemee" target="_blank"><img style="border-radius:70px" src="https://avatars.githubusercontent.com/u/31719240?v=4" width="64" height="64" alt="Dain KIM"></a></p>', unsafe_allow_html=True)
    git_col2.markdown('<p align="center"><a href="https://github.com/Ihyun" target="_blank"><img style="border-radius:70px" src="https://avatars.githubusercontent.com/u/32431157?v=4" width="64" height="64" alt="Ihyun SONG"></a></p>', unsafe_allow_html=True)
    git_col3.markdown('<p align="center"><a href="https://github.com/iamtrueline" target="_blank"><img style="border-radius:70px" src="https://avatars.githubusercontent.com/u/79238023?v=4" width="64" height="64" alt="Jinseon KANG"></a></p>', unsafe_allow_html=True)
    git_col4.markdown('<p align="center"><a href="https://github.com/kimminji2018" target="_blank"><img style="border-radius:70px" src="https://avatars.githubusercontent.com/u/74283190?v=4" width="64" height="64" alt="Minji KIM"></a></p>', unsafe_allow_html=True)
    git_col5.markdown('<p align="center"><a href="https://github.com/NayoungLee-de" target="_blank"><img style="border-radius:70px" src="https://avatars.githubusercontent.com/u/69383548?v=4" width="64" height="64" alt="Nayoung LEE"></a></p>', unsafe_allow_html=True)
    git_col6.markdown('<p align="center"><a href="https://github.com/sw6820" target="_blank"><img style="border-radius:70px" src="https://avatars.githubusercontent.com/u/52646313?v=4" width="64" height="64" alt="Wonji SHIN"></a></p>', unsafe_allow_html=True)