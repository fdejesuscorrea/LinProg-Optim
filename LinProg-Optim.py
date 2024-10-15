# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:11:16 2024

@author: aka_cosi
"""
import tkinter as tk
import sympy as sp
from tkinter import ttk
import tkinter.messagebox as messagebox
import pandas as pd
import numpy as np

global root
global maxz
maxz = None
global restricciones
restricciones=None
global text_A
text_A=None
global text_B
text_B=None
global text_C
text_C=None
global text_simbolos
text_simbolos=None

def interpretar_ecuacion(ec):
    if(ec==None or ec=="" or ec=="\n"):
        return None
    try:
        # Convertir el texto en una expresión SymPy
        ecuacion = sp.sympify(ec)
        return ecuacion
    except (sp.SympifyError, SyntaxError) as e:
        print(f"Error al interpretar la ecuación: {e}")
        return None
def convertir_a_texto(data):
    if isinstance(data, list):
        return '\n'.join(map(str, data))
    return str(data)
def sustituciones(simbolos, valores):
    """
    Crea un diccionario para sustituir símbolos con valores correspondientes.
    
    Argumentos:
    simbolos -- Lista de símbolos de SymPy
    valores -- Lista de valores correspondientes a los símbolos
    
    Retorna:
    Un diccionario donde las claves son los símbolos y los valores son los valores a sustituir.
    """
    if len(simbolos) != len(valores):
        raise ValueError("La cantidad de símbolos y valores debe ser la misma.")
    # Crear el diccionario combinando símbolos y valores
    diccionario_sustituciones = dict(zip(simbolos, valores))
    return diccionario_sustituciones


def interpretar_campos():
    global restricciones
    global maxz
    global numvars
    global text_A
    global text_B
    global text_C
    global text_simbolos
    global root
    numvars=int(entry_numvars.get())
    ecuacion=entry_ecuacion.get()
    maxz = interpretar_ecuacion(ecuacion)
    desigualdades=text_restriccion.get("1.0",tk.END).split("\n")[:-1]
    print(desigualdades)
    restricciones = list(map(interpretar_ecuacion, desigualdades))
    simbolos = list(sp.symbols(f'x1:{numvars+1}'))
    coefs_dict=maxz.as_coefficients_dict()
    x = [var for var in coefs_dict if var != 1]  # Excluimos la constante 1
    C = [coefs_dict[var] for var in x]
    B= list(map(lambda e:e.rhs,restricciones))
    restricder=list(map(lambda d:d.lhs,restricciones))
    variables=simbolos
    matriz_coeficientes = []
    for expresion in restricder:
        # Obtener el diccionario de coeficientes
        coef_dict = expresion.as_coefficients_dict()
        
        # Crear una fila de coeficientes para esta expresión
        fila_coeficientes = []
        for var in variables:
            # Si la variable está en el diccionario, tomar su coeficiente
            # Si no está, agregar 0
            coef = coef_dict.get(var, 0)
            fila_coeficientes.append(coef)
        
        # Agregar la fila de coeficientes a la matriz
        matriz_coeficientes.append(fila_coeficientes)
    A=matriz_coeficientes
    text_A = tk.Text(root, height=5, width=10, state='normal')
    text_B = tk.Text(root, height=5, width=10, state='normal')
    text_C = tk.Text(root, height=5, width=10, state='normal')
    text_simbolos = tk.Text(root, height=5, width=10, state='normal')
    
    # Insertar los datos en los widgets de texto
    text_A.insert(tk.END, convertir_a_texto(A))
    text_B.insert(tk.END, convertir_a_texto(B))
    text_C.insert(tk.END, convertir_a_texto(C))
    text_simbolos.insert(tk.END, convertir_a_texto(simbolos))
    
    # Hacer que los campos de texto no sean editables
    text_A.config(state='disabled')
    text_B.config(state='disabled')
    text_C.config(state='disabled')
    text_simbolos.config(state='disabled')
    
    labelA=tk.Label(root,text="Coeficientes\ntecnologicos")
    labelB=tk.Label(root,text="Recursos\n requerimientos")
    labelC=tk.Label(root,text="Coeficientes\nde costo")
    labelSimbolos=tk.Label(root,text="Variables\nde decisión")
    
    labelA.grid(row=4, column=0, padx=1, pady=10,sticky="E")
    labelSimbolos.grid(row=4, column=1, padx=1, pady=10)
    labelB.grid(row=4, column=2, padx=1, pady=10)
    labelC.grid(row=4, column=3, padx=1, pady=10,sticky="W")
    
    text_A.grid(row=5, column=0, padx=1, pady=10,sticky="E")
    text_simbolos.grid(row=5, column=1, padx=1, pady=10)
    text_B.grid(row=5, column=2, padx=1, pady=10)
    text_C.grid(row=5, column=3, padx=1, pady=10,sticky="W")


    print(x)
    print(C)
    print(B)
    print(A)
    print(restricder)
    print(simbolos)
    print(numvars,maxz,restricciones)


def linProgOpti():
    global root
    root = tk.Tk()
    root.title("Metodo Simplex - Programación Lineal")
    root.geometry("820x600")
    global entry_ecuacion
    global entry_numvars
    global text_restriccion
    

    
    # campos y etiquetas para la introducción de datos
    label_ecuacion = tk.Label(root, text="Inserta la ecuación a maximizar (Max Z):")
    label_ecuacion.grid(row=0, column=0, padx=10, pady=10)
    entry_ecuacion = tk.Entry(root, width=30)
    entry_ecuacion.grid(row=0, column=1, columnspan=2, padx=10, pady=10)
    
    label_numvars = tk.Label(root, text="Inserta el numero de variables que usa tu función:")
    label_numvars.grid(row=1, column=0, padx=10, pady=10)
    entry_numvars = tk.Entry(root, width=30)
    entry_numvars.grid(row=1, column=1, columnspan=2, padx=10, pady=10)
    
    label_restriccion = tk.Label(root, text="Inserta las restricciones:")
    label_restriccion.grid(row=2, column=0, padx=10, pady=10)
    text_restriccion = tk.Text(root, height=10, width=20)
    text_restriccion.grid(row=2, column=3, padx=10, pady=10)
    
    button_interpretar = tk.Button(root, text="Interpretar campos", command=interpretar_campos)
    button_interpretar.grid(row=3, column=2, padx=10, pady=10)
    


    root.mainloop()

if(__name__=="__main__"):
    linProgOpti()