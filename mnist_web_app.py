import pickle
import numpy as np
import streamlit as st
import cv2
#from io import StringIO
#import PIL
import os
#from PIL import Image
import matplotlib
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import streamlit.components.v1 as componentes
import csv

IMG_HEIGHT=28
IMG_WIDTH=28

img_folder=r'.\numeros'
predicciones=[]

def chunks(L, n):
    cadena=""
    #st.write("The numbers predicted are:",end=', ')
    for i in range(0, len(L), n):  
        mylist=L[i:i+n]
        max_index, max_value = max(enumerate(mylist), key=lambda x: x[1])
        predicciones.append(max_index)
        cadena=cadena + str(max_index) + ', '           
    #st.write(cadena)
    return predicciones

def predecir_imagenes():
    st.empty()
    st.header('ALGORITMO DE TENSORFLOW-IDENTIFICACIÓN DE IMAGENES')
    filename=r'.\trainedmodel.sav'
    loaded_model=pickle.load(open(filename,'rb'))
    image_list=[]
    img_predicted=[]
    img_predicted=np.array(img_predicted)
    for file in os.listdir(img_folder):
        image=cv2.imread(os.path.join(img_folder, file),cv2.IMREAD_GRAYSCALE)  
        try:
            image=cv2.resize(image, (IMG_HEIGHT,IMG_WIDTH),interpolation = cv2.INTER_AREA)
        except:
            break
        image=np.array(image)
        image = image.astype('float32')
        image /= 255 
        image_list.append(image)
    image_np_array=np.array(image_list)
    image_np_array=image_np_array.reshape(-1,IMG_HEIGHT*IMG_HEIGHT)
    preds=loaded_model.predict(image_np_array).round().astype(int)
    flat_pred=[item for sublist in preds for item in sublist]
    cargar_imagenes_1(chunks(flat_pred,10),image_list)
    st.write("Predicting finished")
    return image_list

def cargar_imagenes_0():
    for file in os.listdir(img_folder):      
        imagen=mpimg.imread(os.path.join(img_folder, file))     
        st.image(imagen)
        st.write(file)

def cargar_imagenes_1(predicciones,image_list):
    k=1000
    j=2000
    i=0
    for file in os.listdir(img_folder):      
        imagen=mpimg.imread(os.path.join(img_folder, file))
        st.image(imagen)
        st.write(file)
        st.text_input(label="Predicción",key=k,max_chars=1,value=predicciones[i])
        k=k+1
        i=i+1

def cargar_image_loader():
        uploaded_file=st.file_uploader("Introduzca el número de 0-9 a predecir en formato imagen:")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name)
            try:
                with (open(os.path.join(img_folder,uploaded_file.name), 'wb')) as f:
                    image.save(os.path.join(img_folder,uploaded_file.name))
            except:
                raise IOError
            st.write('Fichero guardado')       

def main():
    st.header('ALGORITMO DE TENSORFLOW-IDENTIFICACIÓN DE IMAGENES')
    cargar_imagenes_0()
    cargar_image_loader()
    st.button("Hacer Prediccion",on_click=predecir_imagenes)
    #st.button("Añadir a Modelo",on_click=createModelAdd)

if __name__=='__main__':
    main()

  