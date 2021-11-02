"""
The goal of this file is to take input aerofoil specifications and to create a
3D model of an STL file. This may be done from them specifying the twist,
cord lengths, overall length, etc.

All units are in I'm very sorry to say imperial
################################################################################
#Written by Aaron Schwan
#schwanaaron@gmail.com
################################################################################
"""
#Imports
import numpy as np
import math
import matplotlib.pyplot as plt



#General Functions
def aerofoil_four_digit_NACA(m,p,yt):
    """
    Returns a aerofoil using the 4 digit NACA cambered aerofoil

    m = maximum camber 100*m = first NACA number
    p = location of maximum camber 10*p = second digit in the NACA number
    yt = thickness 100*yt = third and fourth digits in the NACA number
    """
    a0 = 0.2969
    a1 = -0.1260
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1015 #(-0.1036)

    number_of_points = 100
    x_space = np.linspace(0, 1, num=number_of_points)#creates a linear space of 100 items between 0 and 1
    switch_number = round(number_of_points*p) #number that the function will switch at

    #creating the lower surface
    x_lower = np.zeros(number_of_points)
    y_lower = np.zeros(number_of_points)

    #creating the upper surface
    x_upper = np.zeros(number_of_points)
    y_upper = np.zeros(number_of_points)

    for i in range(0,switch_number):
        yc = (m*(2*p*x_space[i]-x_space[i]**2)/(p**2))
        yti = 5*yt*(a0*x_space[i]**(1/2) +a1*x_space[i] +a2*x_space[i]**2 + a3*x_space[i]**3 +a4*x_space[i]**4)
        theta = math.atan(2*m*(p-x_space[i])/p**2)

        #upper section
        x_upper[i] = x_space[i]-yti*math.sin(theta)
        y_upper[i] = yc+yti*math.cos(theta)

        #lower section
        x_lower[i] = x_space[i]+yti*math.sin(theta)
        y_lower[i] = yc-yti*math.cos(theta)

    for i in range(switch_number,number_of_points):
        yc = (m*((1-2*p)+2*p*x_space[i]-x_space[i]**2))/(1-p)**2
        yti = 5*yt*(a0*x_space[i]**(1/2) +a1*x_space[i] +a2*x_space[i]**2 + a3*x_space[i]**3 +a4*x_space[i]**4)
        theta = math.atan(2*m*(p-x_space[i])/(1-p)**2)

        #upper section
        x_upper[i] = x_space[i]-yti*math.sin(theta)
        y_upper[i] = yc+yti*math.cos(theta)

        #lower section
        x_lower[i] = x_space[i]+yti*math.sin(theta)
        y_lower[i] = yc-yti*math.cos(theta)


    #Making an overall coordinate list
    x_vals = np.concatenate((x_upper, x_lower), axis=0)
    y_vals = np.concatenate((y_upper, y_lower), axis=0)

    coords = list(zip(x_vals.T,y_vals.T))
    return coords


#Defining Variables
overall_length = 12;

#Definging Input Functions
def chord_length(z):
    """
    Takes an input of position returns the length at that point
    """
    return 5;

def aerofoil_along_blade(z):
    """
    Takes in position along the turbine and returns a 2D slice of an aerofoil
    """
    aerofoil_slice_temp = aerofoil_four_digit_NACA(0.02, 0.4, .12);
    chord_len = chord_length(1);
    aerofoil_slice = [(elem1*chord_len,elem2) for elem1, elem2 in aerofoil_slice_temp]
    return aerofoil_slice
#Plotting the coords
coords = aerofoil_along_blade(1);
plot_list = [(elem1,elem2) for elem1, elem2 in coords]
plt.scatter(*zip(*plot_list))
plt.show()
