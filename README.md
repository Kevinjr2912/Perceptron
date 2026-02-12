# Perceptron Simple

## Introducción

Uno de los problemas comunes en el campo de la ciencia de datos es la predicción de ciertas variables a partir de otras. Por ejemplo, debemos saber si es conveniente conceder un préstamo a un cliente dependiendo de factores como su edad, sus ingresos mensuales, etc. Para ello, simplemente debemos clasificar cada caso en dos categorías, si concedemos el préstamo (1) o no (0). Por lo tanto, estamos ante un problema de clasificación.

Los problemas de clasificación y regresión constituyen las dos clases principales de lo que se llama aprendizaje automático supervisado (consiste en enseñar a un algoritmo a hacer predicciones. Para conseguirlo, se alimenta con ayuda de datos que ya están etiquetados).

Para entender cómo están relacionadas entre sí las distintas variables y poder emitir una predicción, necesitamos partir de un conjunto de observaciones. Así, posteriormente, será capaz de predecir el precio de una casa conociendo solamente las demás variables.

Para resolver problemas complejos, en las últimas décadas se han desarrollado y popularizado un conjunto de modelos con una estructura común, las conocidas como **redes neuronales artificiales**.

## Conceptos previos
- **Neurona:** Son células nerviosas biológicas interconectadas entre sí que permiten el tratamiento y la transmisión de señales químicas y eléctricas.
- **Dendritas:** Son ramificaciones que reciben la información de otras neuronas. Los núcleos celulares tratan la información recibida a partir de las dendritas. 
- **Núcleos celulares:** Tratan la información recibida a partir de las dendritas.
- **Sinapsis:**: Sirve de conexión entre las neuronas

## Definición

En 1957, Frank Ronsenblant desarrolló un modelo simple de neurona basado en el modelo de McCulloch y Pitts y en una regla de aprendizaje basada en la corrección del error. A este modelo le llamó Perceptrón. Una de las características que más interés despertó de este modelo fue su capacidad de aprender a reconocer patrones.

El Perceptron es una neurona artificial y, por tanto, una unidad de red neuronal, que, en esencia, trata de imitar el funcionamiento de una neurona biológica. Es la unidad fundamental que recibe información, la procesa y emite una señal de salida. Una neurona artificial que toma varias entradas, las pondera y produce una salida binaria (0 o 1). Como se ha de notar, se trata de un algoritmo para el aprendizaje supervisado de clasificadores binarios. 

El Perceptrón es especialmente útil en problemas de clasificación binaria, es decir, problemas en los que se busca clasificar elementos en dos categorías. Sin embargo, su capacidad de generalización es limitada, por lo que no es adecuado para problemas más complejos. A pesar de esto, el Perceptrón sigue siendo una herramienta valiosa en el aprendizaje automático y se utiliza como base para modelos más avanzados, como las redes neuronales multicapa.

## Fuentes de información

- https://blog.damavis.com/perceptron-simple-definicion-matematica-y-propiedades/
- https://gamco.es/glosario/perceptron/
