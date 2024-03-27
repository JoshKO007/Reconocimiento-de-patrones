#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:26:14 2024

@author: joshtincorteslopez
"""

import numpy as np
import random

def Hold_Out(matriz_iris, nombre_archivo):
    with open(nombre_archivo, 'w') as archivo:
        matriz_setosa = matriz_iris[:50,:]
        archivo.write("Iris Setosa\n")
        archivo.write(str(matriz_setosa))
        archivo.write("\n")

        matriz_Versicolor = matriz_iris[50:100,:]
        archivo.write("\nIris Versicolor\n")
        archivo.write(str(matriz_Versicolor))
        archivo.write("\n")

        matriz_Virginica = matriz_iris[100:150,:]
        archivo.write("\nIris Virginica\n")
        archivo.write(str(matriz_Virginica))
        archivo.write("\n")

        while True:
            numero_entero = int(input("Por favor, ingresa un número entre 0 y 100 para definir el porcentaje (R): "))
            if 0 <= numero_entero <= 100:
                archivo.write("\n")
                archivo.write("Porcentaje válido: {}\n".format(numero_entero))
                break
            else:
                archivo.write("El porcentaje ingresado está fuera del rango permitido. Por favor, inténtelo de nuevo.\n")

        porcentaje = int((numero_entero * 50) / 100)
        residuo = int(50 - porcentaje)

        matriz_setosa_cf = matriz_setosa[:porcentaje, :]
        archivo.write("\nIris Setosa C.F\n")
        archivo.write(str(matriz_setosa_cf))
        archivo.write("\n")

        matriz_setosa_cp = matriz_iris[porcentaje:50,:-1]
        archivo.write("\nIris Setosa C.P\n")
        archivo.write(str(matriz_setosa_cp))
        archivo.write("\n")

        matriz_Versicolor_cf = matriz_iris[50:100-residuo,:]
        archivo.write("\nIris Versicolor C.F\n")
        archivo.write(str(matriz_Versicolor_cf))
        archivo.write("\n")

        matriz_Versicolor_cp = matriz_iris[100-residuo:100,:-1]
        archivo.write("\nIris Versicolor C.P\n")
        archivo.write(str(matriz_Versicolor_cp))
        archivo.write("\n")

        matriz_Virginica_cf = matriz_iris[100:150-residuo,:]
        archivo.write("\nIris Virginica C.F\n")
        archivo.write(str(matriz_Virginica_cf))
        archivo.write("\n")

        matriz_Virginica_cp = matriz_iris[150-residuo:150,:-1]
        archivo.write("\nIris Virginica C.P\n")
        archivo.write(str(matriz_Virginica_cp))
        archivo.write("\n")

        archivo.write("\nConcatenacion de valores C.F\n")
        concatenacion_CF = np.concatenate((matriz_setosa_cf, matriz_Versicolor_cf, matriz_Virginica_cf), axis=0)
        archivo.write(str(concatenacion_CF))
        archivo.write("\n")

        archivo.write("\nConcatenacion de valores C.P\n")
        concatenacion_CP = np.concatenate((matriz_setosa_cp, matriz_Versicolor_cp, matriz_Virginica_cp), axis=0)
        archivo.write(str(concatenacion_CP))
        archivo.write("\n")

def Leave_One_Out(matriz_iris, nombre_archivo):
    with open(nombre_archivo, 'w') as archivo:
        for i in range(len(matriz_iris)):
            numeroAleatorio = random.randint(0, len(matriz_iris) - 1)  
            matrizResultado = np.delete(matriz_iris, numeroAleatorio, axis=0)
            archivo.write("Iteración " + str(i + 1) + ":\n")
            archivo.write("Dato de validacion:\n")
            datoValidacion = matriz_iris[numeroAleatorio:numeroAleatorio+1,:-1]
            archivo.write(str(datoValidacion) + '\n\n')
            archivo.write("Datos de entrenamiento:\n")
            archivo.write(str(matrizResultado) + '\n\n')


def K_Fold_Cross_validation(matriz_iris, nombre_archivo):
    K  = int(input("Ingresa el valor de K: "))
    Division = int(len(matriz_iris)/K)

    with open(nombre_archivo, 'w') as archivo:
        for i in range(K):
            Matriz1 = matriz_iris[:Division,:]
            Matriz2 = matriz_iris[Division:Division*2,:]
            Matriz3 = matriz_iris[Division*2:Division*3,:]
            Matriz4 = matriz_iris[Division*3:Division*4,:]
            Matriz5 = matriz_iris[Division*4:Division*5,:]
            Matriz6 = matriz_iris[Division*5:Division*6,:]
            Matriz7 = matriz_iris[Division*6:Division*7,:]
            Matriz8 = matriz_iris[Division*7:Division*8,:]
            Matriz9 = matriz_iris[Division*8:Division*9,:]    
            Matriz10 = matriz_iris[Division*9:Division*10,:]
           
        for i in range(K):
            archivo.write("Iteración {}\n\n".format(i + 1))
            
            if i == 0:
                Matriz_CE = np.concatenate((Matriz2, Matriz3, Matriz4, Matriz5, Matriz6, Matriz7, Matriz8, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz1[:,:-1]
            elif i == 1:
                Matriz_CE = np.concatenate((Matriz1, Matriz3, Matriz4, Matriz5, Matriz6, Matriz7, Matriz8, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz2[:,:-1]
            elif i == 2:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz4, Matriz5, Matriz6, Matriz7, Matriz8, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz3[:,:-1]
            elif i == 3:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz3, Matriz5, Matriz6, Matriz7, Matriz8, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz4[:,:-1]
            elif i == 4:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz3, Matriz4, Matriz6, Matriz7, Matriz8, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz5[:,:-1]
            elif i == 5:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz3, Matriz4, Matriz5, Matriz7, Matriz8, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz6[:,:-1]
            elif i == 6:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz3, Matriz4, Matriz5, Matriz6, Matriz8, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz7[:,:-1]
            elif i == 7:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz3, Matriz4, Matriz5, Matriz6, Matriz7, Matriz9, Matriz10), axis=0)
                Matriz_CP = Matriz8[:,:-1]
            elif i == 8:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz3, Matriz4, Matriz5, Matriz6, Matriz7, Matriz8, Matriz10), axis=0)
                Matriz_CP = Matriz9[:,:-1]
            else:
                Matriz_CE = np.concatenate((Matriz1, Matriz2, Matriz3, Matriz4, Matriz5, Matriz6, Matriz7, Matriz8, Matriz9), axis=0)
                Matriz_CP = Matriz10[:,:-1]

            archivo.write("Matriz de entrenamiento:\n")
            archivo.write(str(Matriz_CE) + '\n\n')

            archivo.write("Matriz de práctica:\n")
            archivo.write(str(Matriz_CP) + '\n\n')