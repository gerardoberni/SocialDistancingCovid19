# SocialDistancingCovid19
# Susana distancia en Android y IOS

## Resumen

Mediante el modelo TinyYoloV3 se implementó una aplicación para Android y IOS que consiste en la identificación en tiempo real de personas y la determinación de si están en riesgo de contraer la enfermedad COVID 19 al encontrarse cerca de otras personas o si se encuentran en una "sana distancia"

## Desarrollo


* Dependiendo del dispositivo que se esta utilizando son las dependencias y las librerias que se usaron por lo que dentro del folder para cada OS se encuentra un README file para cada uno. 


* Una vez que se implementó el modelo exitosamente se procedió a desarrollar la detección de distancia entre las personas detectadas; como referencia nos basamos en el [GitHub](https://github.com/Ank-Cha/Social-Distancing-Analyser-COVID-19?fbclid=IwAR3uywKvB-b3KUExnuVpOrlrOEx5Kb_sURQPDJAGfoEQ7ac4hlKrqF7FIPk) de **Ankush Chaudhari** para implementar el cálculo de la distancia entre las bounding boxes desplegadas en la aplicación para ambos OS nos basamos en el mismo algoritmo.

* Posteriormente se llevó a cabo un proceso de refactorización del programa para mejorar el desempeño de nuestro modelo y remover elementos inecesarios.

## ¿Como usar?

* [Android](https://github.com/gerardoberni/SocialDistancingCovid19/tree/master/SanaDistancia_Android)
* [iOS](https://github.com/gerardoberni/SocialDistancingCovid19/tree/master/Susana-iOS)

## Conclusión

Como resultado se logró implementar un modelo que puede mostrar en pantalla bounding boxes ya sea en color verde (sin riesgo de contagio) o en color rojo (riesgo de contagio) junto con una línea roja que conecta las bounding boxes de las personas en riesgo.

Aún hay área de mejora ya que puede ser aún más preciso y más eficiente pero a nuestro parecer se obtuvo un resultado favorable.

Tambien se podria hacer uso de algun método para medir profundidades.