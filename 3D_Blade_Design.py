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
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
import open3d as o3d


#General Functions
def find_normal_to_plane(A,B,C):
    """
    Finds the normal vector to three point plane
    """
    vec1 = [A[0]-B[0],A[1]-B[1],A[2]-B[2]]
    vec2 = [C[0]-B[0],C[1]-B[1],C[2]-B[2]]

    vec3 = np.cross(vec1,vec2)

    return vec3
def create_mesh_of_aerofoils(X,Y,Z):
    """
    creates a mesh from all 3D points itterating arround in a circle essentially
    """
    #looping conditions
    num_points_per_slice = next((i for i, x in enumerate(Z) if x), Z[0])
    print(num_points_per_slice)
    #num_points = num_points_per_slice+1;#one point so end attaches to start
    #spliting lists into numpy arrays
    x = np.reshape(np.array(X),(-1,num_points_per_slice))
    y = np.reshape(np.array(Y),(-1,num_points_per_slice))
    z = np.reshape(np.array(Z),(-1,num_points_per_slice))

    #Verticies
    vertices = np.array((X,Y,Z)).T

    #creating faces
    num_faces = (num_points_per_slice-1)*(len(x)-1)+1+num_points_per_slice*2-2
    faces = np.zeros((num_faces*2,3));
    counter = 0;
    for i in range(len(x)-1):
        for j in range(num_points_per_slice-1):
            bottom_ind = num_points_per_slice*i+j
            #top triangle
            faces[counter] = [bottom_ind,bottom_ind+1,(bottom_ind+1+num_points_per_slice)]

            counter+=1;
            #bottom triangle
            faces[counter] = [bottom_ind, (bottom_ind+1+num_points_per_slice), (bottom_ind+num_points_per_slice)]
            counter+=1;


    #Bottom Surface
    x_bot = x[0].reshape((-1,int(num_points_per_slice/2)))
    y_bot = y[0].reshape((-1,int(num_points_per_slice/2)))
    z_bot = z[0].reshape((-1,int(num_points_per_slice/2)))

    xc_bot = (x_bot[0]+x_bot[1])/2
    yc_bot = (y_bot[0]+y_bot[1])/2

    vertices_bot = []

    #Tiling bottom and top surfaces###################
    #############################################




    faces = faces.astype(int)

    # Create the mesh
    blade = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            blade.vectors[i][j] = vertices[f[j],:]

    # Write the mesh to file "cube.stl"
    blade.save('blade.stl')



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

    return x_vals,y_vals

def twist(x,y,theta,twist_axis):
    """
    twists all points around the axis
    """
    a = twist_axis[0]
    b = twist_axis[1]
    x_new = (np.array(x)-a)*math.cos(theta)-(np.array(y)-b)*math.sin(theta)-a;
    y_new = (np.array(y)-b)*math.cos(theta)+(np.array(x)-a)*math.sin(theta)-b;
    return x_new, y_new;

#Defining Variables
blade_length = 12*25.4;
m = 0.02
p = 0.4
yt = 0.12
twist_axis = (0.4,0)
twist_angle_deg = 0;
twist_angle_rad = twist_angle_deg*math.pi/180;
num_sec = 2;

#Definging Input Functions
def chord_length(z):
    """
    Takes an input of position returns the length at that point
    """
    return (10*25.4)#-9*z/12);
def twist_along_length(z):
    """
    Define how much twist per increase in z occurs
    """
    return (twist_angle_rad/blade_length)*z

def aerofoil_along_blade(z):
    """
    Takes in position along the turbine and returns a 2D slice of an aerofoil
    """
    #Get base aerofoil
    x_vals,y_vals = aerofoil_four_digit_NACA(m, p, yt);
    #Get scaling Factor
    chord_len = chord_length(z);
    #Twist around an axis and scale
    x_vals,y_vals = twist(x_vals*chord_len,y_vals*chord_len,twist_along_length(z),twist_axis)
    #Add the z coordinates
    z_vals = np.ones(len(x_vals))*z
    return x_vals,y_vals,z_vals

#Creating the aerofoil
z_space = np.linspace(0,blade_length,num_sec)

#iterate through length of blade
x_set = []
y_set = []
z_set = []

for ind,z_sp in enumerate(z_space):

    x,y,z = aerofoil_along_blade(z_sp);
    length_vals = len(x);
    #appending to overall list

    x_set[ind*length_vals:length_vals*ind+length_vals-1] = x;
    y_set[ind*length_vals:length_vals*ind+length_vals-1] = y;
    z_set[ind*length_vals:length_vals*ind+length_vals-1] = z;


create_mesh_of_aerofoils(x_set,y_set,z_set)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(np.array(x_set), np.array(y_set), np.array(z_set))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
