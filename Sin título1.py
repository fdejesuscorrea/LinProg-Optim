# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:04:18 2024

@author: admin
"""

import sympy as sp

# Definir 10 símbolos x1, x2, ..., x10
num_simbolos = 10
simbolos = sp.symbols(f'x1:{num_simbolos+1}')

print(simbolos)