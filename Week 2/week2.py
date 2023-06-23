import matplotlib.pyplot as plt
import numpy as np
import random

# Self-defined functions
import sys
import os

def linear_interpolation(a, b, t):
    return (1-t)*a + t*b

def distance_euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def triangle_area_2D(A, B, C):
    return 0.5 * np.abs(np.cross(A-B, A-C))

def slope_2D(A, B):
    return (B[1]-A[1])/B[0]-A[0]

class Triangle_Area_Bisection_2D:
    def __init__(self, vertices, p):
        if(len(vertices) < 3):
            A = np.array([6,9])
            B = np.array([10,14])
            C = np.array([8,28])
            self.vertices = np.array([A, B, C])
        else:
            self.vertices = vertices
        self.t = []
        self.G = (vertices[0] + vertices[1] + vertices[2]) / 3
        self.p = p
        self.P = []
        self.Q = []
        # The thrid vertex of small triangle part
        self.top = []
        # The (unit) direction of the dicision boundary
        self.d = []
        
    def ft(self, p1, p2, p3, p4, t):
        terms = self.compute_quadratic_coefficient(p1, p2, p3, p4)
        return terms[0] * pow(t, 2) + terms[1]*t + terms[2]
    
    def compute_quadratic_coefficient(self, p1, p2, p3, p4):
        quadratic = (p1[0]*p4[1]-p2[0]*p4[1]+p2[0]*p1[1]-p4[0]*p1[1]+p4[0]*p2[1]-p1[0]*p2[1])
        linear = (p1[0]*p2[1]+p3[0]*p1[1]-p3[0]*p2[1]-p1[0]*p3[1]+p2[0]*p3[1]-p2[0]*p1[1])/2
        constant = (p3[0]*p4[1]-p3[0]*p1[1]-p1[0]*p4[1]-p4[0]*p3[1]+p4[0]*p1[1]+p1[0]*p3[1])/2
        return np.array([quadratic, linear, constant])
    
    def compute_quadratic_delta(self, terms):
        return pow(terms[1], 2) - 4 * terms[0] * terms[2]
    
    def compute_t(self, terms):
        delta = self.compute_quadratic_delta(terms)
#         print("delta: ", delta)
        t1 = (-terms[1]+pow(delta, 0.5)) / 2 / terms[0]
        t2 = (-terms[1]-pow(delta, 0.5)) / 2 / terms[0]
        # print("t1: ", t1, "t2: ", t2)
        
        if(t1 >= 1/2 - 1e-6 and t1< 1 - 1e-6):
            return t1
        if(t2 >= 1/2 - 1e-6 and t2< 1 - 1e-6):
            return t2
        
        return -1
    
    def solve_quadratic(self, p1, p2, p3):
        terms = self.compute_quadratic_coefficient(p1, p2, p3, self.p)
        t_temp = self.compute_t(terms)
        self.t = self.t + [t_temp]
        return t_temp
    
    def get_t(self):
        if(self.ft(self.vertices[0], self.vertices[1], self.vertices[2], self.p, 0.5)*self.ft(self.vertices[0], self.vertices[1], self.vertices[2], self.p, 1) <= 1e-6):
            t_temp = self.solve_quadratic(self.vertices[0], self.vertices[1], self.vertices[2])
            P_temp = linear_interpolation(self.vertices[0], self.vertices[1], t_temp)
            Q_temp = linear_interpolation(self.vertices[0], self.vertices[2], 1/2/t_temp)
            d_temp = (P_temp - self.p) / np.linalg.norm(P_temp - self.p)
            print("1", P_temp, " ", self.p, " ", d_temp, "t", t_temp)
            self.P = self.P + [P_temp]
            self.Q = self.Q + [Q_temp]
            self.d = self.d + [d_temp]
            self.top = self.top + [self.vertices[0]]
            
        if(self.ft(self.vertices[1], self.vertices[2], self.vertices[0], self.p, 0.5)*self.ft(self.vertices[1], self.vertices[2], self.vertices[0], self.p, 1) <= 1e-6):
            t_temp = self.solve_quadratic(self.vertices[1], self.vertices[2], self.vertices[0])
            P_temp = linear_interpolation(self.vertices[1], self.vertices[2], t_temp)
            Q_temp = linear_interpolation(self.vertices[1], self.vertices[0], 1/2/t_temp)
            d_temp = (P_temp - self.p) / np.linalg.norm(P_temp - self.p)
            print("2", P_temp, " ", self.p, " ", d_temp, "t", t_temp)
            self.P = self.P + [P_temp]
            self.Q = self.Q + [Q_temp]
            self.d = self.d + [d_temp]
            self.top = self.top + [self.vertices[1]]
            
        if(self.ft(self.vertices[2], self.vertices[0], self.vertices[1], self.p, 0.5)*self.ft(self.vertices[2], self.vertices[0], self.vertices[1], self.p, 1) <= 1e-6):
            t_temp = self.solve_quadratic(self.vertices[2], self.vertices[0], self.vertices[1])
            P_temp = linear_interpolation(self.vertices[2], self.vertices[0], t_temp)
            Q_temp = linear_interpolation(self.vertices[2], self.vertices[1], 1/2/t_temp)
            d_temp = (P_temp - self.p) / np.linalg.norm(P_temp - self.p)
            print("3", P_temp, " ", self.p, " ", d_temp, "t", t_temp)
            self.P = self.P + [P_temp]
            self.Q = self.Q + [Q_temp]
            self.d = self.d + [d_temp]
            self.top = self.top + [self.vertices[2]]
            
    def v_decision_boundary(self):
        for i in range(len(self.P)):
            v_line_2D(self.p, self.P[i])
            v_line_2D(self.p, self.Q[i])
            
    def v_triangle(self):
        v_triangle_2D(self.vertices[0], self.vertices[1], self.vertices[2])
        
    def v_vertices(self):
        plt.plot(self.vertices[0][0], self.vertices[0][1], 'o', color='r')
        plt.plot(self.vertices[1][0], self.vertices[1][1], 'o', color='r')
        plt.plot(self.vertices[2][0], self.vertices[2][1], 'o', color='r')
        
    def v_result(self):
        self.v_vertices()
        plt.plot(self.p[0], self.p[1], 'o', color='r')
        self.v_triangle()
        v_triangle_barycentre(self.vertices[0], self.vertices[1], self.vertices[2])
        self.get_t()
        self.v_decision_boundary()
        
    def print_info(self):
        print("The coordinates of point P: ", self.P)
        print("The coordinates of point Q: ", self.Q)
        print("The area of triangle: ", triangle_area_2D(self.vertices[0], self.vertices[1], self.vertices[2]))
        print("The area of small half-area triangle: ", triangle_area_2D(self.P[0], self.Q[0], self.top[0]))

class opt_gradient_descent:
    def __init__(self, num_iterations = 1000, lr=0.05):
        self.t1 = random.random()
        self.t2 = random.random()
        self.lr = lr
        self.num_iterations = num_iterations
    
    def loss(self):
        return np.power(self.t1*self.t2 - 1/2, 2)
    
    def gradients(self):
        dt1 = 2 * np.power(self.t2, 2) * self.t1 - self.t2
        dt2 = 2 * np.power(self.t1, 2) * self.t2 - self.t1
        return dt1, dt2
        
    def update_parameters(self, dt1, dt2):
        self.t1 = self.t1 - self.lr * dt1
        self.t2 = self.t2 - self.lr * dt2
    
    def train_one_round(self):
        loss = self.loss()
        dt1, dt2 = self.gradients()
        self.update_parameters(dt1, dt2)
        print("t1 = {}, t2 = {}, loss = {}".format(self.t1, self.t2, loss))

    def train(self):
        for i in range(self.num_iterations):
            self.train_one_round()
    
    def reset(self):
        self.t1 = random.random()
        self.t2 = random.random()

def get_line(p1, p2):
    d = p2 - p1
    t = np.linspace(0, 1, 1000)
    x = p1[0] + d[0] * t
    y = p1[1] + d[1] * t
    return x, y

def v_line_parametric_2D(p,d):
    t = np.linspace(0, 1, 1000)
    x = p[0] + d[0] * t
    y = p[1] + d[1] * t
    plt.plot(x, y, 'b-')
    # return x, y
    plt.xlabel('x')
    plt.ylabel('y')
    
def v_line_2D(p1, p2):
    d = p2 - p1
    v_line_parametric_2D(p1, d)

    plt.xlabel('x')
    plt.ylabel('y')
    return d

def v_triangle_2D(p1,p2,p3):
    v_line_2D(p1,p2)
    v_line_2D(p2,p3)
    v_line_2D(p1,p3)
    
def v_triangle_barycentre(p1,p2,p3):
    G = (p1 + p2 + p3) / 3
    plt.plot(G[0], G[1], 'o', color='r')
    return G

fig, ax = plt.subplots()
ax.set_xlim([0, 10])  # Setting limit for better visualization
ax.set_ylim([0, 10])  # Setting limit for better visualization
points, = ax.plot([], [], marker='o', linestyle='')
boundary, = ax.plot([], [], 'b-')
GD=opt_gradient_descent()

def onclick(event):
    x, y = event.xdata, event.ydata
    current_points = points.get_data()
    if len(current_points[0]) < 4:
        new_points_x = list(current_points[0]) + [x]
        new_points_y = list(current_points[1]) + [y]
        points.set_data(new_points_x, new_points_y)

        if(len(new_points_x) == 3):
            v_triangle_2D(np.array([new_points_x[0], new_points_y[0]]), 
                          np.array([new_points_x[1], new_points_y[1]]), 
                          np.array([new_points_x[2], new_points_y[2]]))
            G = v_triangle_barycentre(np.array([new_points_x[0], new_points_y[0]]), np.array([new_points_x[1], new_points_y[1]]), np.array([new_points_x[2], new_points_y[2]]))
        
        if(len(new_points_x) == 4):
            new_p = np.array([new_points_x[3], new_points_y[3]])
            vertices = np.array([np.array([new_points_x[0], new_points_y[0]]), np.array([new_points_x[1], new_points_y[1]]), np.array([new_points_x[2], new_points_y[2]])])
            problem = Triangle_Area_Bisection_2D(vertices, new_p)
            problem.v_result()
    fig.canvas.draw()

def on_keypress(event):
    if event.key == 'i':
        print("Initialize!")
        current_points = points.get_data()
        boundary.set_data([], [])
        GD.reset()
        # print(list(current_points[0]), " ", list(current_points[1]))
        P = linear_interpolation(np.array([current_points[0][0], current_points[1][0]]), np.array([current_points[0][1], current_points[1][1]]), GD.t1)
        Q = linear_interpolation(np.array([current_points[0][0], current_points[1][0]]), np.array([current_points[0][2], current_points[1][2]]), GD.t2)
        # print(P, " ", Q)
        temp_x, temp_y = get_line(P, Q)
        boundary.set_data(temp_x, temp_y)

    elif event.key == 't':
        print("Train!")
        GD.train_one_round()
        current_points = points.get_data()
        P = linear_interpolation(np.array([current_points[0][0], current_points[1][0]]), np.array([current_points[0][1], current_points[1][1]]), GD.t1)
        Q = linear_interpolation(np.array([current_points[0][0], current_points[1][0]]), np.array([current_points[0][2], current_points[1][2]]), GD.t2)
        # print(P, " ", Q)
        temp_x, temp_y = get_line(P, Q)
        boundary.set_data(temp_x, temp_y)
    
    elif event.key == 'c':
        print("Clean!")
        points.set_data([],[])

    else:
        print(f"You pressed {event.key}.")
    
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_keypress)

plt.show()
