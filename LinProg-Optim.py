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

global maxz
maxz = None
global restricciones
restricciones=None

def interpretar_ecuacion(ec):
    if(ec==None or ec=="" or ec=="\n"):
        return None
    try:
        # Convertir el texto en una expresión SymPy
        ecuacion = sp.sympify(ec)
        print(f"Ecuación interpretada: {ecuacion}")
        return ecuacion
    except (sp.SympifyError, SyntaxError) as e:
        print(f"Error al interpretar la ecuación: {e}")
        return None

def interpretar_campos():
    global restricciones
    global maxz
    global numvars
    numvars=int(entry_numvars.get())
    ecuacion=entry_ecuacion.get()
    maxz = interpretar_ecuacion(ecuacion)
    desigualdades=text_restriccion.get("1.0",tk.END).split("\n")[:-1]
    print(desigualdades)
    restricciones = list(map(interpretar_ecuacion, desigualdades))
    print(numvars,maxz,restricciones)


def linProgOpti():
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
    text_restriccion = tk.Text(root, height=10, width=50)
    text_restriccion.grid(row=2, column=3, padx=10, pady=10)
    
    button_interpretar = tk.Button(root, text="Interpretar campos", command=interpretar_campos)
    button_interpretar.grid(row=3, column=2, padx=10, pady=10)


    root.mainloop()

if(__name__=="__main__"):
    linProgOpti()