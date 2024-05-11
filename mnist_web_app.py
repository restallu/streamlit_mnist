import pickle
import numpy as np
import streamlit as st
import cv2
from io import StringIO
import PIL
import os
from PIL import Image
import streamlit as st

img_folder=r'.\numeros'

def cargar_imagenes():
    for imagen in os.listdir(img_folder):
        st.image(imagen)

def cargar_image_loader():
    uploaded_file=st.file_uploader("Introduzca imagen a escanear:")
    if uploaded_file is not None:
        datos=uploaded_file.getvalue()
        st.write(datos)
        pickle.dump(datos,open(os.path.join(img_folder,uploaded_file,'wb')))
    
def load_model():
    filename=r'C:\Datasets\mnist\trainedmodel.sav'
    loaded_model=pickle.load(open(filename,'rb'))

def main():
    st.title('ALGORITMO DE TENSORFLOW-IDENTIFICACIÓN DE IMAGENES')
    cargar_imagenes()
    cargar_image_loader()
    

if __name__=='__main__':
    main()

  