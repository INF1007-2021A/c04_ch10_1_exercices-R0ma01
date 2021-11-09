#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import cmath
import matplotlib.pyplot as plt
import random
import scipy.integrate as integ
import math

"""
Fonction qui retourne une liste de num nombre d'éléments
également répartis entre start et stop
"""
def linear_values(start: int, stop : int, num : int) -> np.ndarray:
    return np.linspace(start, stop, num)

"""
D'une liste de coordonnées cartésiennes on tire 
les cordonnées polaires sous forme (rayon, angle)
Pour aider la compréhension, les formules sont:
angle = np.arctan(y/x) donne un angle en radiants.
rayon = np.sqrt(x**2+y**2) donne un float.
"""
def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:

    return np.array([(np.sqrt(x**2+y**2),np.arctan(y/x)) for x,y in cartesian_coordinates])
    #ou
    #return np.array([cmath.polar(c) for c in cartesian_coordinates])

#le array donné sera de forme différente though

"""
fonction qui peremets de trouver l'indexe la plus proche d'un nombre fournis
"""
def find_closest_index(values: np.ndarray, number: float) -> int:
      return (np.abs(values - number).argmin())

def f(x):
    return ((x**2)*np.sin(1/(x**2))+x)

def utiliser_matplot():
    points = np.linspace(-1,1,250)
    y = np.array([f(x) for x in points])
    plt.plot(points, y)
    plt.show()

"""
Fonction qui permets de calculer la valeur de "pi" en calculant le nombre de points dans une surface 
délimitée par un arc de cercle
"""
def monte_carlo(nombre: int) -> float:
    point_x = []
    point_y = []
    proportion = 0

    for i in range(nombre):
        point_x.append(random.uniform(0,1))
        point_y.append(random.uniform(0, 1))
        if ((point_x[i])**2+(point_y[i])**2) <1:
            plt.plot(point_x[i], point_y[i], "bo")
            proportion +=1
        else:
            plt.plot(point_x[i], point_y[i], "ro")

    estimation = 4*proportion/nombre

    plt.show()

    return estimation

def integrande(x):
    return math.exp(-(x**2))

"""
Permets d'évaluer l'intégrale sur l'interval donnée en argument
"""
def evaluer_integrale(minim : int, maxim : int):

    return integ.quad(integrande,minim, maxim)

"""
Permets d'afficher l'intégrale de la fonction sur l'interval donné en argument
"""
def afficher_integrale(minim: int, maxim: int):
    for i in range(minim, maxim):
        plt.plot(i,integrande(i),"ro")

    plt.show()


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values(-1.3,2.5,64))

    print(coordinate_conversion(np.array([[1,2],[3,4]])))

    print(find_closest_index(np.array([1,2,3,4,5]),5))

    utiliser_matplot()

    print(monte_carlo(2000))

    print(evaluer_integrale(-(math.inf),math.inf ))

    print(evaluer_integrale(-4,4))

    afficher_integrale(-4,4)

