#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:01:24 2024

@author: joshtincorteslopez
"""
import numpy as np

import random

def cargar_iris(ruta_archivo):
  matriz_iris = np.loadtxt(ruta_archivo, delimiter=',', dtype=str)
  return matriz_iris

def Hold_Out(matriz_iris):    
    matriz_setosa = matriz_iris[:50,:]
    print("")
    print("Iris Setosa")
    print(matriz_setosa)
    print("")

    matriz_Versicolor = matriz_iris[50:100,:]
    print("")
    print("Iris Versicolor")
    print(matriz_Versicolor)
    print("")

    matriz_Virginica = matriz_iris[100:150,:]
    print("")
    print("Iris Virginica")
    print(matriz_Virginica)
    print("")


    while True:
        numero_entero = int(input("Por favor, ingresa un número entre 0 y 100 para definir el porcentaje (R): "))
        if 0 <= numero_entero <= 100:
            print("Porcentaje válido:", numero_entero)
            break  # Rompe el bucle si se ingresa un número válido
        else:
            print("El porcentaje ingresado está fuera del rango permitido. Por favor, inténtelo de nuevo.")


    porcentaje = int((numero_entero * 50) / 100)
    residuo = int(50 - porcentaje)

    matriz_setosa_cf = matriz_setosa[:porcentaje, :]
    print("")
    print("Iris Setosa C.F")
    print(matriz_setosa_cf)
    print("")

    matriz_setosa_cp = matriz_iris[porcentaje:50,:-1]
    print("")
    print("Iris Setosa C.P")
    print(matriz_setosa_cp)
    print("")

    matriz_Versicolor_cf = matriz_iris[50:100-residuo,:]
    print("")
    print("Iris Versicolor C.F")
    print(matriz_Versicolor_cf)
    print("")

    matriz_Versicolor_cp = matriz_iris[100-residuo:100,:-1]
    print("")
    print("Iris Versicolor C.P")
    print(matriz_Versicolor_cp)
    print("")

    matriz_Virginica_cf = matriz_iris[100:150-residuo,:]
    print("")
    print("Iris Virginica C.F")
    print(matriz_Virginica_cf)
    print("")

    matriz_Virginica_cp = matriz_iris[150-residuo:150,:-1]
    print("")
    print("Iris Virginica C.P")
    print(matriz_Virginica_cp)
    print("")

    print("Concatenacion de valores C.F")
    concatenacion_CF = np.concatenate((matriz_setosa_cf, matriz_Versicolor_cf, matriz_Virginica_cf),axis = 0)
    print("")
    print (concatenacion_CF)
    print("")

    print("Concatenacion de valores C.P")
    concatenacion_CP = np.concatenate((matriz_setosa_cp, matriz_Versicolor_cp, matriz_Virginica_cp),axis = 0)
    print("")
    print (concatenacion_CP)
    print("")

def Leave_One_Out(matriz_iris):
    for i in range(len(matriz_iris)):
        numeroAleatorio = random.randint(0, len(matriz_iris) - 1)  
        matrizResultado = np.delete(matriz_iris, numeroAleatorio, axis=0)
        print("Iteración" , i + 1 , ":")
        print ("Dato de validacion: ")
        datoValidacion = matriz_iris[numeroAleatorio:numeroAleatorio+1,:-1]
        print(datoValidacion)
        print(" ")
        print("Datos de entrenamiento: ")
        print(matrizResultado)
        print(" ")
        print(" ")

def K_Fold_Cross_validation(matriz_iris):
    K  = int(input("Ingresa el valor de K: "))
    Division = int(len(matriz_iris)/K)

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
      print("Iteración", i+1)
      print("")
      
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

      print("Matriz de entrenamiento:")
      print(Matriz_CE)
      print("")
      print("Matriz de practica:")
      print(Matriz_CP)
      print("")

        