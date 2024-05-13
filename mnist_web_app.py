import pickle
import numpy as np
import streamlit as st
import cv2
import PIL
import os
from PIL import Image
import matplotlib
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from streamlit_js_eval import streamlit_js_eval

IMG_HEIGHT=28
IMG_WIDTH=28

img_folder=r'./numeros'
predicciones=[]

# Devuelve los valores de las prediccinones. El valor de la predicción corresponde
# a la posición de la neurona que presenta el maximo valor de las 10 neuronas de salida

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

# En esta función se desserializa el modelo y se aplica a las muestras de validación

def predecir_imagenes():
    st.empty()
    st.header('ALGORITMO DE TENSORFLOW-IDENTIFICACIÓN DE IMAGENES')
    filename=r'./trainedmodel.sav'
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
    cargar_imagenes_1(chunks(flat_pred,10))
    st.write("Predicting finished")
    return image_list

# Carga o presentación inicial de imágenes

def cargar_imagenes_0():
    for file in os.listdir(img_folder):      
        imagen=mpimg.imread(os.path.join(img_folder, file))     
        st.image(imagen)
        st.write(file)

# Carga de imágenes con predicción hecha

def cargar_imagenes_1(predicciones):
    k=1000
    i=0
    for file in os.listdir(img_folder):      
        imagen=mpimg.imread(os.path.join(img_folder, file))
        st.image(imagen)
        st.write(file)
        st.text_input(label="Predicción",key=k,max_chars=1,value=predicciones[i])
        k=k+1
        i=i+1
    cargar_image_loader()
    st.button("Hacer Prediccion",on_click=predecir_imagenes)
    for i in range(20):
        st.write(' ')

#Logica asociada al widget de adición de nuevas imágenes

def cargar_image_loader():
        uploaded_file=st.file_uploader("Puede añadir sus propias muestras para predecir. Seran reducidas a 28x28 pixels:")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name)
            try:
                with (open(os.path.join(img_folder,uploaded_file.name), 'wb')) as f:
                    image.save(os.path.join(img_folder,uploaded_file.name))
            except:
                raise IOError
            st.write('Fichero guardado')  
            streamlit_js_eval(js_expressions="parent.window.location.reload()")   
            
def main():
    
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
    cargar_imagenes_0()
    cargar_image_loader()
    st.button("Hacer Prediccion",on_click=predecir_imagenes)

if __name__=='__main__':
    main()

  
