#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:12:49 2024

@author: joshtincorteslopez
"""

import funciones
import funcion_texto

ruta_archivo = "/Users/joshtincorteslopez/Desktop/iris/iris.csv"
matriz_iris = funciones.cargar_iris(ruta_archivo)

print("Instrucciones: ")
print("")
print("1. Imprimir el dataset sin modificaciones")
print("2. Funciones de separación imprimiendo en la consola (No saldran los datos completos en algunos casos)")
print("3. Funciones de separacion escritas en un archivo de texto plano (Los datos se escriben completos")
print("")
print("Escribe el número de la instruccion que le quieres dar al programa: ")

instruccion = input()

if instruccion == '1':
    print("")
    print("Impresión del dataset sin modificaciones")
    print("")
    print(matriz_iris)
elif instruccion == '2':
    print("")
    print("Opciones: ")
    print("1. Hold-out")
    print("2. Leave-one-Out")
    print("3. K-Fold-Cross-Validation")
    print("")
    print("Escribe el numero de la opcion que quieras seleccionar: ")
    opcion = input()
    if opcion == '1':
        funciones.Hold_Out(matriz_iris)
    elif opcion == '2':
        funciones.Leave_One_Out(matriz_iris)
    elif opcion == '3':
        funciones.K_Fold_Cross_validation(matriz_iris)
elif instruccion == '3':
    print("")
    print("Opciones: ")
    print("1. Hold-out")
    print("2. Leave-one-Out")
    print("3. K-Fold-Cross-Validation")
    print("")
    print("Escribe el numero de la opcion que quieras seleccionar: ")
    opcion = input()
    if opcion == '1':
         nombre_archivo = "Hold_out"
         funcion_texto.Hold_Out(matriz_iris, nombre_archivo)
         print("El resultado se ha guardado en la carpeta donde está este proyecto")
         print("Abre el archivo desde tu administrador (Aún no se abrirlo automaticamente)")
    elif opcion == '2':
         nombre_archivo = "Leave_One_Out"
         funcion_texto.Leave_One_Out(matriz_iris, nombre_archivo)
         print("El resultado se ha guardado en la carpeta donde está este proyecto")
         print("Abre el archivo desde tu administrador (Aún no se abrirlo automaticamente)")
    elif opcion == '3':
         nombre_archivo = "K_Fold_Cross_validation"
         funcion_texto.K_Fold_Cross_validation(matriz_iris, nombre_archivo)
         print("El resultado se ha guardado en la carpeta donde está este proyecto")
         print("Abre el archivo desde tu administrador (Aún no se abrirlo automaticamente)")
        
    