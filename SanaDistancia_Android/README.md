# SocialDistancingCovid19
# Susana distancia en Android

## Resumen

Mediante el modelo TinyYoloV3 se implementó una aplicación para Android que consiste en la identificación en tiempo real de personas y la determinación de si están en riesgo de contraer la enfermedad COVID 19 al encontrarse cerca de otras personas o si se encuentran en una "sana distancia"

## Desarrollo


* Se utilizo openCv para poder implementar una red neuronal como es TinyYolo haciendo uso de la configuracion y los pesos ya pre entreados. 

* Para guiarnos se tomó como referencia el siguiente [repositorio](https://github.com/ivangrov/Android-Deep-Learning-with-OpenCV) de Ivangrov en donde realiza la implementacion de TinyYolo v3 en un celular Android

* Una vez que se implementó el modelo exitosamente se procedió a desarrollar la detección de distancia entre las personas detectadas; como referencia nos basamos en el [GitHub](https://github.com/Ank-Cha/Social-Distancing-Analyser-COVID-19?fbclid=IwAR3uywKvB-b3KUExnuVpOrlrOEx5Kb_sURQPDJAGfoEQ7ac4hlKrqF7FIPk) de **Ankush Chaudhari** para implementar el cálculo de la distancia entre las bounding boxes desplegadas en la aplicación.

* Posteriormente se llevó a cabo un proceso de refactorización del programa para mejorar el desempeño de nuestro modelo y remover elementos inecesarios.

## ¿Como usar?

*Para poder hacer uso de este proyecto se uso Android Studio 4.0. Se creo un proyecto para que corriera como minimo en el SDK 21 y como objetivo SDK 28. Es importante tomar en cuenta esto debido a que hay ciertos metodos que en versiones mayores que se volvieron obsoletos. 

*Ya que se tiene esto es necesario poder contar con la libreria de OpenCV version 3.4.5. El repositorio ya cuenta con ella por lo que no deberia de ser ningun problema. Sin embargo, si llegara a haber problemas hay que bajar la libreria de la pagina de OpenCV y hacerla modulo de la aplicacion que estamos usando. 

*Otro de los requisitos que ocuparemos para poder correr esta aplicacion es contar con los archivos de configuracion y los pesos de la red neuronal dentro de nuestro celular. Podemos lograr esto descargando los archivos de la pagina de [DARKNET](https://pjreddie.com/darknet/yolo/) y poniendolos en la siguiente direccion dentro de nuestra SD "/dnns/yolov3-tiny.cfg" y "/dnns/yolov3-tiny.weights".

*Ya con todo esto podemos proceder a cargar la aplicacion a nuestro celular. Para poder realizar esto tenemos que asegurarnos que nuestro dispositivo este en modo desarrollador y que los permisos para poder instalar nuestra propia aplicacion esten activados. Si ya lo estan solo es cuestion de correr la aplicacion en nuestro celular y ya esta listo.

## Conclusión

Como resultado se logró implementar un modelo que puede mostrar en pantalla bounding boxes ya sea en color verde (sin riesgo de contagio) o en color rojo (riesgo de contagio) junto con una línea roja que conecta las bounding boxes de las personas en riesgo.

Aún hay área de mejora ya que puede ser aún más preciso y más eficiente pero a nuestro parecer se obtuvo un resultado favorable.

Tambien se podria hacer uso de algun método para medir profundidades.