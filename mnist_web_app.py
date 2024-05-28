import streamlit as st
import os
import cv2
import pickle
import numpy as np
import shutil
import random
from streamlit_js_eval import streamlit_js_eval
from typing import TypedDict

class Sample(TypedDict):
    i: np.ubyte
    imagen: np.ndarray


path="./numeros/"
img_folder="./numeros/"
IMG_HEIGHT=28
IMG_WIDTH=28

def cargar_home():

    st.write("INICIO PROGRAMA")
    st.sidebar.image('nist.png',use_column_width=False,)
    st.sidebar.text('La base de datos MNIST por sus' )
    st.sidebar.text('siglas en inglés, Modified National') 
    st.sidebar.text('Institute of Standards and Technology') 
    st.sidebar.text('es una extensa colección de base de')
    st.sidebar.text('datos que se utiliza ampliamente')
    st.sidebar.text( 'para el entrenamiento de sistemas')
    st.sidebar.text(' de procesamiento de imágenes.')

    st.image('tf.png',use_column_width=False)
    st.header('ALGORITMO DE TENSORFLOW')
    st.subheader('Identificación de dígitos manuscritos')
    st.write('A continuación usamos el algoritmo TensorFlow para predecir,')
    st.write('numeros manuscritos prodedentes de una colección de MNIST de')
    st.write('60.000 muestras procedentes de la escritura de 300 personas. ')
    st.write('A continuación se muestran algunas imágenes. ')
    st.write('Pulse sobre HACER PREDICCION')
          
def callbackfunct(u,image,p,f):
    elementToAddToModel:Sample={"i":np.ubyte(u), "imagen":image}
    filename="./corrmod/dict"+str(random.randint(0, 99))+".sav"
    pickle.dump(elementToAddToModel,open(filename,'wb'))
    setFileProcessed(f)


def setFileProcessed(fichero):
    shutil.move("./numeros/"+fichero, "./processed")
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def chunks(L, n):
    predicciones=[]
    for i in range(0, len(L), n):  
        mylist=L[i:i+n]
        max_index, max_value = max(enumerate(mylist), key=lambda x: x[1])
        predicciones.append(max_index)       
    return predicciones


def is_directory_empty(path):
    if os.path.exists(path) and os.path.isdir(path):
        if not os.listdir(path):
            return True
        else:
            return False
        
def predecir_imagenes():
    #st.image(cv2.imread("tf.png"))
    #st.header('ALGORITMO DE TENSORFLOW-IDENTIFICACIÓN DE IMAGENES')
    filename=r'./trainedmodel.sav'
    loaded_model=pickle.load(open(filename,'rb'))
    file_list=[]
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
        file_list.append(file)
    image_np_array=np.array(image_list)
    image_np_array=image_np_array.reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)
    preds=loaded_model.predict(image_np_array).round().astype(int)
    flat_pred=[item for sublist in preds for item in sublist]
    #cargar_imagenes_1(chunks(flat_pred,10),image_list)
    cargar_predicciones(chunks(flat_pred,10),image_list,file_list)
    #st.button("Añadir a Modelo",on_click=addToModel)
    st.write("Predicting finished")
    return image_list

def cargar_predicciones(predicciones,image_list,file_list):
    i=0
    cargar_home()
    for image,file in zip(image_list,file_list):
            col1,col2=st.columns([1,1])
            with col1:
                st.image(image)
                st.write(file)
            with col2:
                submitted2=st.button("Next",key=f"Button-{str(i)}")
                if submitted2:
                    setFileProcessed(file)
            with st.form (f"MyForm-{str(i)}"):
                st.text_input("Predicción Realizada:",value=predicciones[i],disabled=True)
                user_input=st.text_input("Predicción Correcta:")
                submitted=st.form_submit_button("Añadir a modelo")
                if submitted:
                    callbackfunct(user_input,image,predicciones[i],file)
            i=i+1

def presentar_imagenes():
    cargar_home()
    col1,col2=st.columns([1,1])
    for file in os.listdir(path):
        img=cv2.imread(os.path.join(path,file),cv2.IMREAD_GRAYSCALE)      
        with col1: 
            st.image(img)
        with col2: 
            st.write(file)
    submitted=st.button("Predecir Set")
    if submitted:
        predecir_imagenes()
def main():
    
    predecir_imagenes()




if __name__=='__main__':
    main()
