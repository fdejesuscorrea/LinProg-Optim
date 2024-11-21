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

global root,my_text,text_C,maxz,restricciones,text_A,text_B,text_simbolos,new_window,desigualdades
maxz = None
my_text="Resultados Por Iteracion Metodo Simplex Revisado\n"
restricciones=None
text_A=None
text_B=None
text_C=None
text_simbolos=None
global b,A,simbolos,z,XB,CB,s,x,xB,Z,C
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

def open_new_window():
    # Crear una nueva ventana
    global new_window,root,my_text
    new_window = tk.Toplevel(root)
    new_window.title("Nueva Ventana")
    new_window.geometry("400x600")
    
    # Crear un marco para contener el canvas y el scrollbar
    frame = ttk.Frame(new_window)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Crear un canvas
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Agregar un scrollbar al canvas
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Crear un frame para el contenido dentro del canvas
    content_frame = ttk.Frame(canvas)
    
    # Configurar el contenido para que sea desplazable
    canvas.create_window((0, 0), window=content_frame, anchor="nw")
    content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    # Mostrar el texto dentro del contenido
    label = ttk.Label(content_frame, text=my_text, wraplength=380, anchor="center", justify="left")
    label.pack(pady=10, padx=10)

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

###############################################################sxcfsdsdsdfsdfsd
from scipy.optimize import linprog
from matplotlib.collections import PolyCollection
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
    x, y = simbolos[0], simbolos[1]

    # Crear una figura para los gráficos
    fig, ax = plt.subplots(figsize=(8, 8))

    # Inicializar listas para las intersecciones de las restricciones
    puntos = []

    # Procesar cada desigualdad
    for desigualdad in desigualdades:
        expr = desigualdad.lhs - desigualdad.rhs
        y_expr = solve(expr, y)

        if y_expr:
            # Crear una función lambda para calcular y en función de x
            y_func = lambdify(x, y_expr[0], 'numpy')
            x_vals = np.linspace(x_lim[0], x_lim[1], 400)
            
            # Verificar si el resultado es un valor escalar o un array
            y_vals = y_func(x_vals)
            if np.isscalar(y_vals):
                y_vals = np.full_like(x_vals, y_vals)  # Repetir el valor escalar en el array
            
            # Evitar valores fuera de los límites de la gráfica
            mask = (y_vals >= y_lim[0]) & (y_vals <= y_lim[1])
            x_vals, y_vals = x_vals[mask], y_vals[mask]
            
            # Trazar la línea de la restricción
            ax.plot(x_vals, y_vals, label=str(desigualdad))
            # Determinar intersecciones con otras restricciones
            for otra in desigualdades:
                if otra != desigualdad:
                    interseccion = solve([expr, otra.lhs - otra.rhs], (x, y))
                    if interseccion:
                        interseccion = {var: float(val) for var, val in interseccion.items()}
                        puntos.append((interseccion[x], interseccion[y]))

    # Filtrar puntos que estén dentro de la región factible
    puntos_factibles = [
        punto for punto in puntos 
        if all((desigualdad.lhs.subs({x: punto[0], y: punto[1]}) <= desigualdad.rhs) for desigualdad in desigualdades)
    ]

    # Sombrear la región factible
    if puntos_factibles:
        puntos_factibles = np.array(puntos_factibles)
        region = plt.Polygon(puntos_factibles, color='gray', alpha=0.3)
        ax.add_patch(region)

    # Resolver el problema de optimización
    c = [-funcion_objetivo.coeff(x), -funcion_objetivo.coeff(y)]  # Maximizar se transforma en minimizar (-)
    A = []
    b = []

    for desigualdad in desigualdades:
        A.append([desigualdad.lhs.coeff(x), desigualdad.lhs.coeff(y)])
        b.append(desigualdad.rhs)

    res = linprog(c, A_ub=A, b_ub=b, bounds=[x_lim, y_lim])

    if res.success:
        optimo = res.x
        ax.scatter(optimo[0], optimo[1], color='red', label=f"Solución óptima: {optimo}")

    # Configurar la gráfica
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Método gráfico para programación lineal')
    ax.legend()
    plt.grid(True)

    plt.show()

def agregar_nueva_variable(B, CB):
    coeficiente_nueva_variable = 1/2
    coeficientes_tecnologicos_nuevos = sp.Matrix([2, 1])

    print(f"\nAgregaremos una nueva variable.\n\n"
          f"nueva funcion objetivo:\n Zmax = 10*x1 + 14*x2 + 0.5*x3\n\n"
          f"Nuevas restricciones:\n4*x1 + 6*x2 + 2*x3<= 24\n2*x1 + 6*x2 +x1 <= 20\n\n")

    resultado = [sp.nsimplify(x) for x in (B * coeficientes_tecnologicos_nuevos)]
    resultado_final = coeficiente_nueva_variable - (CB.T * sp.Matrix(resultado))[0]
    print(f"Nuevo factor para C_j - Z_j: {resultado_final}")
    if resultado_final > 0:
        print("La solucion dejó de ser optima")
    else:
        print("La solucion sigue siendo optima")

def analisis_de_sensibilidad(b,C,A,Binv):
    """
    Realiza el análisis de sensibilidad y muestra los resultados en tablas.

    Parámetros:
    - z: Expresión sympy de la función objetivo.
    - desigualdades: Lista de desigualdades sympy con las restricciones.
    - x_lim, y_lim: Límites para graficar las restricciones.
    """
    d = sp.symbols("d")
    b = sp.Matrix(b)
    Binv = sp.Matrix(Binv)
    for i in range(len(restricciones)):
        restricciones_aux = b
        restricciones_aux[i] += d
        resultado_fracciones = sp.Matrix(Binv * restricciones_aux)
        inequalities = [sp.Ge(x, 0) for x in resultado_fracciones]
        solucion = sp.solve(inequalities, d)
        print(f"\nIntervalo de d para X{i+1}\n")
        print(solucion)

    agregar_nueva_variable(Binv, sp.Matrix(C))


def is_optime(Binv, CB):
    global z, xB, maxz
    # Variables no básicas
    ps = list(set(z) - set(xB))
    psi = [z.index(elemento) for elemento in ps]
    
    # Coeficientes de costos reducidos
    ci = [maxz.coeff(str(elemento)) for elemento in ps]
    pi = A[:, psi]
    
    # Cálculo del vector de costos reducidos
    ret = np.matmul(CB, np.matmul(Binv, pi)) - ci
    
    opt = True
    mini = np.inf
    cand = ""
    
    for r in ret:
        if r < 0:  # Solo seleccionamos las variables con costos negativos
            opt = False
        if r < mini:  # Regla de Bland: Seleccionar el índice menor en caso de empate
            mini = r
    
    if not opt:
        cand = ps[list(ret).index(mini)]  # Índice de la variable que entra
    return opt, cand, ret


def is_factible(cand, XB, Binv):
    # Vector columna de la variable que entra
    pxi = A[:, z.index(cand)]
    alfa = np.matmul(Binv, pxi)
    
    # Calcular θ (solo considerar valores positivos en alfa)
    theta = np.full_like(XB, np.nan, dtype=float)
    theta[alfa > 0] = XB[alfa > 0] / alfa[alfa > 0]
    
    # Si no hay valores positivos, el problema es ilimitado
    if np.all(np.isnan(theta)):
        print("Problema ilimitado: no hay solución factible.")
        return None, None
    
    # Encontrar el mínimo de θ
    mini = np.nanmin(theta)
    i = list(theta).index(mini)
    sale = xB[i]
    return None, sale


def resolver_simplexrev():
    global z,b, xB, maxz, b, Z,C, CB, XB, x, s, A, desigualdades,my_text
    B = np.ones((len(s), len(s)))
    Binv = B
    opti = False
    z = x + s
    xB = s
    Z = np.zeros(len(xB))
    A = np.hstack((A, np.eye(len(s))))
    A = A.astype(float)
    
    iteraciones = 0
    MAX_ITER = 1000  # Límite de iteraciones para prevenir bucles infinitos
    
    while not opti:
        if iteraciones > MAX_ITER:
            raise ValueError("El algoritmo no converge después de demasiadas iteraciones")
        iteraciones += 1
        my_text+="-_-_-_-_-_-_-_-_\n"
        my_text+=("iteración: "+ str(iteraciones)+"\n")
        B = getB(A, z, xB)
        Binv = getBinv(B)
        XB = getXB(Binv, b)
        CB = getCB(xB, maxz)
        Z = getZ(CB, XB)
        opti, entra, ret = is_optime(Binv, CB)
        my_text+=("B: "+ str(B)+"\n")
        my_text+=("Binv: "+ str(Binv)+"\n")
        my_text+=("XB: "+ str(XB)+"\n")
        my_text+=("CB: "+ str(CB)+"\n")
        my_text+=("Z: "+ str(Z)+"\n")
        my_text+=("Opti: "+ str(opti)+"\n")
        my_text+=("entra: "+ str(entra)+"\n")
        my_text+=("xB: "+ str(xB)+"\n")
        my_text+=("z: "+str(z)+"\n")
        if opti:
            my_text+=("-_-_-_-_-_-_-_-_\n-_-_-_-_-_-_-_-_\nsolución:"+str(XB))
            break
        fact, sale = is_factible(entra, XB, Binv)
        print("sale: ", sale)
        if sale is None:
            raise ValueError("Problema ilimitado: no hay solución factible")
        xB[list(xB).index(sale)] = entra
    open_new_window()
    metodo_grafico_programacion_lineal(maxz, restricciones, x_lim=(0, 10), y_lim=(0, 10))
    #b=recursos A=Coefs tecnolo C=coefs de costo Binv=(matriz de holgura)
    analisis_de_sensibilidad(b,C,A,Binv)


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