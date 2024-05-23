#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 2023
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: vides
"""

from RNLFinancialSystemID import RNLFinancialSystemID
w0,D_input,D_output,r = RNLFinancialSystemID(1,[1,2],[0])
w0,D_input,D_output,r = RNLFinancialSystemID(2,[1,2],[0])
w0,D_input,D_output,r = RNLFinancialSystemID(3,list(set(range(15)).difference({0})),[0],filtering=True)