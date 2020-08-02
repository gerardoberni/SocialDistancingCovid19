# Susana distancia en iOS

## Resumen

Mediante el modelo TinyYolo se implementó una aplicación para iOS que consiste en la identificación en tiempo real de personas y la determinación de si están en riesgo de contraer la enfermedad al encontrarse cerca de otras personas o si se encuentran en una "sana distancia"

## Desarrollo

* Se utilizó el framework MLCore de iOS para implementar un modelo de TinyYolo con los weights pre entrenados dentro de los cuales se encuentra la clase "person".

* Para guiarnos se tomó como referencia el siguiente [blogpost](http://machinethink.net/blog/yolo-coreml-versus-mps-graph/) escrito por **Matthijs Hollemans** al igual que el [Github del blog](https://github.com/hollance/YOLO-CoreML-MPSNNGraph/tree/master) en el cual se explica la implementación de TinyYolo en iOS con MLCore.

* Una vez que se implementó el modelo exitosamente se procedió a desarrollar la detección de distancia entre las personas detectadas; como referencia nos basamos en el [GitHub](https://github.com/Ank-Cha/Social-Distancing-Analyser-COVID-19?fbclid=IwAR3uywKvB-b3KUExnuVpOrlrOEx5Kb_sURQPDJAGfoEQ7ac4hlKrqF7FIPk) de **Ankush Chaudhari** para implementar el cálculo de la distancia entre las bounding boxes desplegadas en la aplicación.

* Posteriormente se llevó a cabo un proceso de refactorización del programa para mejorar el desempeño de nuestro modelo y remover elementos inecesarios.

## Cómo usar

* Descargar el proyecto y abrir el archivo .xcodeproj con Xcode y correrlo con un dispositivo iOS con versión 11 o superior del sistema operativo.

## Conclusión

Como resultado se logró implementar un modelo que puede mostrar en pantalla 5 bounding boxes ya sea en color verde (sin riesgo de contagio) o en color rojo (riesgo de contagio) junto con una línea roja que conecta las bounding boxes de las personas en riesgo.

Aún hay área de mejora ya que puede ser aún más preciso y más eficiente pero a nuestro parecer se obtuvo un resultado favorable.
