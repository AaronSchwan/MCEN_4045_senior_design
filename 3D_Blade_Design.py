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
    num_points_per_slice = 200#next((i for i, x in enumerate(Z) if x), Z[0])

    #num_points = num_points_per_slice+1;#one point so end attaches to start
    #spliting lists into numpy arrays
    x = np.reshape(np.array(X),(-1,num_points_per_slice))
    y = np.reshape(np.array(Y),(-1,num_points_per_slice))
    z = np.reshape(np.array(Z),(-1,num_points_per_slice))

    x_new = np.zeros((x.shape[0],x.shape[1]+1))
    y_new = np.zeros((y.shape[0],y.shape[1]+1))
    z_new = np.zeros((z.shape[0],z.shape[1]+1))
    for ind,x_num in enumerate(x):
        x_new[ind][0:-1] = x_num
        x_new[ind][-1] = x_num[0]
    for ind,y_num in enumerate(y):
        y_new[ind][0:-1] = y_num
        y_new[ind][-1] = y_num[0]
    for ind,z_num in enumerate(z):
        z_new[ind][0:-1] = z_num
        z_new[ind][-1] = z_num[0]


    #Defining Lists to add to
    m = len(x_new[0])*len(x_new)-2
    tb_vert = len(x_new[0])*8*2+50000
    tb_faces = 4*len(x_new[0])*2+50000
    faces = np.zeros((m+tb_faces,3))
    vertices = np.zeros((m*2+tb_vert,3))

    counter_vert = 0;
    counter_face = 0;

    #itterating and appending to lists
    for i in range(len(x_new)-1):
        #itterate through levels
        for j in range(len(x_new[0])-1):
            #Bottom left corner
            bl_x = x_new[i][j]
            bl_y= y_new[i][j]
            bl_z = z_new[i][j]

            #Bottom right corner
            br_x = x_new[i][j+1]
            br_y = y_new[i][j+1]
            br_z = z_new[i][j+1]

            #Top left corner
            tl_x = x_new[i+1][j]
            tl_y= y_new[i+1][j]
            tl_z = z_new[i+1][j]

            #Top right corner
            tr_x = x_new[i+1][j+1]
            tr_y = y_new[i+1][j+1]
            tr_z = z_new[i+1][j+1]

            #Vertices
            vertices[counter_vert] = [bl_x,bl_y,bl_z]
            counter_vert+=1;
            vertices[counter_vert] = [br_x,br_y,br_z]
            counter_vert+=1;
            vertices[counter_vert] = [tl_x,tl_y,tl_z]
            counter_vert+=1;
            vertices[counter_vert] = [tr_x,tr_y,tr_z]
            counter_vert+=1;

            #Faces
            faces[counter_face] = [counter_vert-2,counter_vert-3,counter_vert-4]
            counter_face+=1
            faces[counter_face] = [counter_vert-2,counter_vert-1,counter_vert-3]
            counter_face+=1

    #Meshing for top surface
    mid = int(len(x_new[0])/2)
    zt = z_new[-1][-1]
    xc = np.flip(np.array(x_new[-1][mid:]))
    yc = np.flip(np.array(y_new[-1][mid:]))


    #Upper half
    for i in range(len(xc)-1):

            #Bottom left corner
            bl_x = x_new[-1][i]
            bl_y= y_new[-1][i]

            #Bottom right corner
            br_x = x_new[-1][i+1]
            br_y = y_new[-1][i+1]

            #Top left corner
            tl_x = xc[i]
            tl_y= yc[i]

            #Top right corner
            tr_x = xc[i+1]
            tr_y = yc[i+1]

            #Vertices
            vertices[counter_vert] = [bl_x,bl_y,zt]
            counter_vert+=1;
            vertices[counter_vert] = [br_x,br_y,zt]
            counter_vert+=1;
            vertices[counter_vert] = [tl_x,tl_y,zt]
            counter_vert+=1;
            vertices[counter_vert] = [tr_x,tr_y,zt]
            counter_vert+=1;

            #Faces
            faces[counter_face] = [counter_vert-2,counter_vert-3,counter_vert-4]
            counter_face+=1
            faces[counter_face] = [counter_vert-2,counter_vert-1,counter_vert-3]
            counter_face+=1

    #Meshing for Bottom surface
    mid = int(len(x_new[0])/2)
    zt = z_new[0][-1]
    xc = np.flip(np.array(x_new[0][mid:]))
    yc = np.flip(np.array(y_new[0][mid:]))

    #Upper half
    for i in range(len(xc)-1):

            #Bottom left corner
            bl_x = x_new[0][i]
            bl_y= y_new[0][i]

            #Bottom right corner
            br_x = x_new[0][i+1]
            br_y = y_new[0][i+1]

            #Top left corner
            tl_x = xc[i]
            tl_y= yc[i]

            #Top right corner
            tr_x = xc[i+1]
            tr_y = yc[i+1]

            #Vertices
            vertices[counter_vert] = [bl_x,bl_y,zt]
            counter_vert+=1;
            vertices[counter_vert] = [br_x,br_y,zt]
            counter_vert+=1;
            vertices[counter_vert] = [tl_x,tl_y,zt]
            counter_vert+=1;
            vertices[counter_vert] = [tr_x,tr_y,zt]
            counter_vert+=1;

            #Faces
            faces[counter_face] = [counter_vert-4,counter_vert-3,counter_vert-2]
            counter_face+=1
            faces[counter_face] = [counter_vert-3,counter_vert-1,counter_vert-2]
            counter_face+=1

    #converting to numpy arrays
    faces = np.array(faces)
    vertices = np.array(vertices)

    #converting indices to proper integer form
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
    x_vals = np.concatenate((x_upper, np.flip(x_lower)), axis=0)
    y_vals = np.concatenate((y_upper, np.flip(y_lower)), axis=0)

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
blade_length = 8*25.4;
blade_start = 0.0125*1000

m = 0.04
p =  0.4
yt =.12
twist_angle_deg = 40;
twist_angle_rad = twist_angle_deg*math.pi/180;
num_sec = 100;
RPM = 2000
air_speed = 10
alpha = 14

#Definging Input Functions
def chord_length(z):
    """
    Takes an input of position returns the length at that point
    """
    Re = 50000
    rho = 1.225
    visc = 1.789*10**(-5)
    tan_vel = z*RPM*2*math.pi/60/1000
    mag_vel = (air_speed**2+tan_vel**2)**(1/2)

    #c = Re*visc/(mag_vel*rho)*1000
    TSR = (z/1000*RPM*2*math.pi/60)/air_speed
    phi = 2*math.atan(1/(TSR))/3
    c = 8*math.pi*z*(1-math.cos(phi))/(3*1.4)

    return c;
def twist_along_length(z):
    """
    Define how much twist per increase in z occurs
    """
    Re = 50000
    rho = 1.225
    visc = 1.789*10**(-5)
    tan_vel = z*RPM*2*math.pi/60/1000
    mag_vel = (air_speed**2+tan_vel**2)**(1/2)
    a = alpha*math.pi/180

    #theta = math.atan(tan_vel/air_speed)+a
    TSR = (z/1000*RPM*2*math.pi/60)/air_speed
    phi = 2*math.atan(1/(TSR))/3
    theta = phi - alpha*math.pi/180

    return theta

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
z_space = np.linspace(blade_start,blade_length,num_sec)

#iterate through length of blade
x_set = []
y_set = []
z_set = []
c = chord_length(z_space[0])
twist_axis = (c/4,0)
print(twist_axis)
blade_twist_list = []
blade_chord_list = []
print(z_space)

for ind,z_sp in enumerate(z_space):

    #x,y,z = aerofoil_along_blade(z_sp);
    #length_vals = len(x);
    #appending to overall list
    blade_twist_list.append(twist_along_length(z_sp))
    blade_chord_list.append(chord_length(z_sp))
    """
    x_set[ind*length_vals:length_vals*ind+length_vals-1] = x;
    y_set[ind*length_vals:length_vals*ind+length_vals-1] = y;
    z_set[ind*length_vals:length_vals*ind+length_vals-1] = z;
    """

#create_mesh_of_aerofoils(x_set,y_set,z_set)
print(blade_twist_list)
print(blade_chord_list)
x_vals,y_vals = aerofoil_four_digit_NACA(0.04, 0.4, 0.12);

#plt.plot(x_vals,y_vals,linewidth=7.0)
#plt.plot([-0.1,1],[0,0],c='black',linewidth=3.0)
#plt.plot([-0.1,1],[0.1,0],c='black',linewidth=3.0)
#plt.plot(z_space,blade_twist_list,c='black',linewidth=3.0)
plt.plot(z_space,blade_chord_list,c='black',linewidth=3.0)
plt.xlabel("Radius (mm)")
plt.ylabel("Chord Diameter (mm)")
plt.show()
