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
from sympy import symbols, Eq, lambdify
import matplotlib.pyplot as plt
from sympy import symbols, Eq, lambdify,solve
from sympy.solvers.inequalities import reduce_rational_inequalities

global root,text_C,maxz,restricciones,text_A,text_B,text_simbolos,new_window,desigualdades
maxz = None
restricciones=None
text_A=None
text_B=None
text_C=None
text_simbolos=None
global b,A,simbolos,z,XB,CB,s,x,xB,Z
global Aith,Bith,XBith

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
    global restricciones,maxz,numvars,text_A,text_B,text_C,text_simbolos,root,desigualdades
    global B,Binv,b,A,C,simbolos,s,x
    numvars=int(entry_numvars.get())
    ecuacion=entry_ecuacion.get()
    maxz = interpretar_ecuacion(ecuacion)
    desigualdades=text_restriccion.get("1.0",tk.END).split("\n")[:-1]
    restricciones = list(map(interpretar_ecuacion, desigualdades))
    simbolos = list(sp.symbols(f'x1:{numvars+1}'))
    coefs_dict=maxz.as_coefficients_dict()
    x = simbolos#[var for var in coefs_dict if var != 1]  # Excluimos la constante 1
    numholg=len(restricciones)
    s= list(sp.symbols(f's1:{numholg+1}'))
    C = [coefs_dict[var] for var in x]
    b= list(map(lambda e:e.rhs,restricciones))
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
    mostrar_ecuacion()


def mostrar_ecuacion():
    text_A = tk.Text(root, height=5, width=10, state='normal')
    text_B = tk.Text(root, height=5, width=10, state='normal')
    text_C = tk.Text(root, height=5, width=10, state='normal')
    text_simbolos = tk.Text(root, height=5, width=10, state='normal')
    
    # Insertar los datos en los widgets de texto
    text_A.insert(tk.END, convertir_a_texto(A))
    text_B.insert(tk.END, convertir_a_texto(b))
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

def open_window():
    global new_window
    new_window = tk.Toplevel(root)
    new_window.title("Resultados del Método Simplex")
    new_window.geometry("400x600")

def getCB(xB,maxz):
    CB=[]
    for i in xB:
        CB.append(maxz.coeff(i))
    return CB

def getB(A,z,xB):
    indices = []
    for elemento in xB:
        if elemento in z:
            indices.append(z.index(elemento))
    B=A[:,indices]
    return B
def getBinv(B):
    Binv=np.linalg.inv(B)
    return Binv
def getXB(Binv,b):
    XB=np.matmul(Binv,b)
    return XB
def getZ(CB,XB):
    print(CB,XB)
    Z=np.matmul(CB,XB)
    return Z
def is_optime(Binv,CB):
    global z,xB,maxz
    ps=list(set(z)-set(xB))
    psi=[]
    for elemento in ps:
        if elemento in z:
            psi.append(z.index(elemento))
    ci=[]
    pi=A[:,psi]
    for elemento in ps:
        if elemento in ps:
            ci.append(maxz.coeff(str(elemento)))

    ret=np.matmul(CB,np.matmul(Binv,pi))-ci

    opt=True
    mini=np.inf
    cand=""
    print("ret: ",ret)
    for r in ret:
        if(r<=0):
            opt=False
        if(r<mini):
            mini=r
    cand=z[list(ret).index(mini)]
    return opt,cand,ret
    
def is_factible(cand,XB,Binv):
    i=cand
    pxi=A[:,z.index(cand)]
    alfa=np.matmul(Binv,pxi)
    theta = np.zeros_like(XB, dtype=float)  # Inicializar con ceros
    # Dividir solo cuando alfa es mayor que cero
    theta[alfa > 0] = XB[alfa > 0] / alfa[alfa > 0]
    # Los elementos donde alfa es cero o menor se mantienen como NaN
    theta[alfa <= 0] = np.nan
    mini=min(theta)
    i=list(theta).index(mini)
    sale=xB[i]
    return None,sale

def metodo_grafico_programacion_lineal(funcion_objetivo, desigualdades, x_lim=(0, 10), y_lim=(0, 10)):
    """
    Método gráfico para problemas de programación lineal.
    
    Parámetros:
    - funcion_objetivo: expresión de sympy para la función objetivo a maximizar.
    - desigualdades: lista de expresiones sympy con las restricciones (desigualdades).
    - x_lim: tupla que define el rango de valores de x para la gráfica (ej. (0, 10)).
    - y_lim: tupla que define el rango de valores de y para la gráfica (ej. (0, 10)).
    """
    global simbolos
    x,y=simbolos[0],simbolos[1]
    # Crear una figura para los gráficos
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plotear cada desigualdad como una línea en el gráfico
    for desigualdad in desigualdades:
        # Convertir la desigualdad en una ecuación y resolver para y en términos de x
        expr = desigualdad.lhs - desigualdad.rhs
        y_expr = solve(expr, y)
        
        if y_expr:
            # Crear una función lambda de la expresión de y en función de x
            y_func = lambdify(x, y_expr[0], 'numpy')
            x_vals = np.linspace(x_lim[0], x_lim[1], 400)
            y_vals = np.full_like(x_vals, y_func(x_vals)) if np.isscalar(y_func(x_vals)) else y_func(x_vals)


            # Trazar la línea de la restricción
            ax.plot(x_vals, y_vals, label=str(desigualdad))
            
            # Rellenar el área factible según el tipo de desigualdad
            if '<=' in str(desigualdad):
                ax.fill_between(x_vals, y_vals, y_lim[0], color='gray', alpha=0.3)
            elif '>=' in str(desigualdad):
                ax.fill_between(x_vals, y_vals, y_lim[1], color='gray', alpha=0.3)
    
    # Configurar el gráfico y añadir etiquetas
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Método gráfico para programación lineal')
    ax.legend()
    plt.grid(True)

    # Convertir la función objetivo en una función evaluable y trazar contornos
    f_obj = lambdify((x, y), funcion_objetivo, 'numpy')
    x_vals = np.linspace(x_lim[0], x_lim[1], 400)
    y_vals = np.linspace(y_lim[0], y_lim[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_obj(X, Y)
    ax.contour(X, Y, Z, levels=20, cmap="RdYlBu", alpha=0.5)

    # Mostrar el gráfico
    plt.show()


def resolver_simplexrev():
    #global new_window,xB,maxz,XB,Z,xB,z
    global z,xB,maxz,b,Z,CB,XB,x,s,A,desigualdades
    B=np.ones((len(s),len(s)))
    Binv=B
    opti=False
    z=x+s
    xB=s
    Z=np.zeros(len(xB))
    A=np.hstack((A,np.eye(len(s))))   
    A=A.astype(float)
    while(not opti):                                                              
        print("-_-_-_-_-_-_-_-_")
        print("iteración")
        B=getB(A,z,xB)
        Binv=getBinv(B)
        XB=getXB(Binv,b)
        CB=getCB(xB,maxz)
        Z=getZ(CB,XB)
        opti,entra,ret=is_optime(Binv,CB)
        fact,sale=is_factible(entra,XB,Binv)
        xB[list(xB).index(sale)]=entra
        print("B: ",B)
        print("Binv: ",Binv)
        print("XB: ",XB)
        print("CB: ",CB)
        print("Z: ",Z)
        print("Opti: ",opti)
        print("entra: ",entra)
        print("sale: ",sale)
        print("xB: ",xB)
        print("z: ",z)
    metodo_grafico_programacion_lineal(maxz, restricciones, x_lim=(0, 10), y_lim=(0, 10))

    #open_window()
'''
def resolver_simplexrev():
    global new_window,xB,maxz,XB,Z,xB,z
    initial_transform()
    opt,entra,ret=is_optime()
    while(not opt):                                                              
        xB[list(xB).index(sale)]=entra
        print("-_-_-_-_-_-_-_-_")
        print("iteración")
        getXB()
        getCB()
        getB()
        getBinv()
        getZ()
        opt,entra,ret=is_optime()

    print(xB,XB,ret)
    #open_window()
'''


def setInterface():
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

    button_interpretar = tk.Button(root, text="resolver", command=resolver_simplexrev)
    button_interpretar.grid(row=3, column=3, padx=10, pady=10)



    

if(__name__=="__main__"):
    setInterface()
    root.mainloop()