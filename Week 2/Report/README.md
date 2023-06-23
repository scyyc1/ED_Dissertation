# Week 2: Linearly Seperate the Area of a Triangle
**Target**: Find a line that bisector the area of a triangle, passing through a fixed point

## Table of Contents
- [0 - Packages and Tools](#0)
- [1 - Triangle Area Bisection Line Passing throught a Fixed Point](#1)
    - [1.1 - Problem Representation](#1-1)
    - [1.2 - Method](#1-2)
    - [1.3 - Result](#1-3)
    - [1.4 - Vertices Classification](#1-3)
    - [1.5 - \*Interactive Plot](#1-5)
- [2 - Reference](#2)

<a name='0'></a>
## 0. Package and Tools


```python
# Packages
import numpy as np
import random
import matplotlib.pyplot as plt

# Self-defined functions
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from util import util
```

<a name='1'></a>
## 1. Triangle Area Bisection Line Passing throught a Fixed Point

<a name='1-1'></a>
### 1.1 Problem Representation 

Following the last week's work, here I use a point on edge on the target triangle and a random point to define the area bisection line (b-line for short).

#### 1.1.1 Conditions
The coordinates of the three vertices of a triangle are A$(x_1,x_2)$, B$(x_2,y_2)$, and C$(x_3,y_3)$. Assume that P is a point on the edge AB. R$(x_4,x_4)$ is a random point that does not collapse with P. Line $PR$ will bisetor the area of $\triangle_{ABC}$.


```python
# Example
A = np.array([1,1])
B = np.array([2,2])
C = np.array([2,-2])
R = np.array([1,0])
p0 = np.array([0,0])

# A = np.array([4,2])
# B = np.array([1,9])
# C = np.array([10,1])
    
util.v_triangle_2D(A,B,C)
plt.plot(A[0], A[1], 'o', color='r')
plt.plot(B[0], B[1], 'o', color='r')
plt.plot(C[0], C[1], 'o', color='r')
plt.plot(R[0], R[1], 'o', color='r')
G = util.v_triangle_barycentre(A,B,C)
```


    
![png](output_5_0.png)
    


<a name='1-2'></a>
### 1.2 Method

I first focus on the situation of vertex $A$.

Suppose that $P$ is on edge $AB$ and b-line $PR$ intersect $\triangle_{ABC}$ with Q on edge $AC$. Following the parameterization method last week, the coordinate of $PQ$ could be represented as

$$P = (1 - t_1)A + tB = \left((1 - t_1)x_1 + tx_2, (1 - t_1)y_1 + ty_2\right)$$ 
$$Q = (1 - t_2)A + tC = \left((1 - t_2)x_1 + tx_3, (1 - t_2)y_1 + ty_3\right)$$

Where

$$t_1, t_2 \in [0, 1]$$ 

and

$$ t_1t_2 = \frac{1}{2}$$

so

$$ \frac{1}{2t_1} = t_2 \in [0,1] $$

gives

$$t_1, t_2 \in [\frac{1}{2}, 1]$$

As points $PQR$ are collinear

$$ m_{PQ} = m_{PR} $$

gives

$$\frac{y_Q - y_P}{x_Q - x_P}=\frac{y_R - y_P}{x_R - x_P}$$

$$
\frac{((1-t_2)y_1 + t_2y_3) - ((1-t_1)y_1 + t_1y_2)}{((1-t_2)x_1 + t_2x_3) - ((1-t_1)x_1 + t_1x_2)} = \frac{y_4 - ((1-t_1)y_1 + t_1y_2)}{x_4- ((1-t_1)x_1 + t_1x_2)}
$$

Substitue $t_1$ with $t$ and $t_2 = \frac{1}{2t_1}$:

$$
\frac{(y_1-y_2)t^2 + \frac{1}{2}(y_3-y_1)}{(x_1-x_2)t^2 + \frac{1}{2}(x_3-x_1)} = \frac{(y_1-y_2)t + (y_4-y_1)}{(x_1-x_2)t + (x_4-x_1)} 
$$

Expend:

$$
LHS = (x_1-x_2)(y_1-y_2)t^3 + (x_1-x_2)(y_4-y_1)t^2 + \frac{1}{2}(x_3-x_1)(y_1-y_2)t + \frac{1}{2}(x_1+x_3)(y_4-y_1)
$$

$$
RHS = (x_1-x_2)(y_1-y_2)t^3 + (x_4-x_1)(y_1-y_2)t^2 + \frac{1}{2}(x_1-x_2)(y_3-y_1)t + \frac{1}{2}(x_4+x_1)(y_1-y_3)
$$

Merge homogeneous terms:

$$
LHS = (x_1y_4+x_2y_1-x_2y_4)t^2 + \frac{1}{2}(-x_1y_2+x_3y_1-x_3y_2)t + \frac{1}{2}(x_1y_4-x_3y_1+x_3y_4)
$$

$$
RHS = (x_1y_2+x_4y_1-x_4y_2)t^2 + \frac{1}{2}(x_1y_3-x_2y_1-x_2y_3) + \frac{1}{2}(-x_1y_3+x_4y_1+x_4y_3)
$$

$$
LHS - RHS = 0
$$

Finally

$$
(x_1y_4-x_2y_4+x_2y_1-x_4y_1+x_4y_2-x_1y_2)t^2+\frac{1}{2}(x_1y_2+x_3y_1-x_3y_2-x_1y_3+x_2y_3-x_2y_1)t+\frac{1}{2}(x_3y_4-x_3y_1-x_1y_4-x_4y_3+x_4y_1+x_1y_3)=0
$$

This is a quadratic function about t.


```python
# Calculate f(t)
def ft(self, p1, p2, p3, p4, t):
    terms = self.compute_quadratic_coefficient(p1, p2, p3, p4)
    return terms[0] * pow(t, 2) + terms[1]*t + terms[2]

def compute_quadratic_coefficient(self, p1, p2, p3, p4):
    quadratic = (p1[0]*p4[1]-p2[0]*p4[1]+p2[0]*p1[1]-p4[0]*p1[1]+p4[0]*p2[1]-p1[0]*p2[1])
    linear = (p1[0]*p2[1]+p3[0]*p1[1]-p3[0]*p2[1]-p1[0]*p3[1]+p2[0]*p3[1]-p2[0]*p1[1])/2
    constant = (p3[0]*p4[1]-p3[0]*p1[1]-p1[0]*p4[1]-p4[0]*p3[1]+p4[0]*p1[1]+p1[0]*p3[1])/2
    return np.array([quadratic, linear, constant])
```

To solve the equation, let's make the it becomes $f(t)=0, t \in [\frac{1}{2}, 1]$. Since only one b-line can be found that passes through a fixed point on a triangle [\[1\]](#ref-1), which means for $f(t)$ there is excatly one solution in $[\frac{1}{2}, 1]$. This should present that 

$$f(\frac{1}{2})*f(1)<=0$$. 

Since here I only consider point $P$ on edge $AB$, there exist three situations. The conclusion can be used to preliminarily determine where the $P$ is on.

<a name='1-3'></a>
### 1.3 Result

This is the complete code of the target problem


```python
class Triangle_Area_Bisection_2D:
    def __init__(self, vertices=np.array([np.array([6,9]), np.array([10,14]), np.array([8,28])]), p):
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
        self.o = []
        
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
        print("delta: ", delta)
        t1 = (-terms[1]+pow(delta, 0.5)) / 2 / terms[0]
        t2 = (-terms[1]-pow(delta, 0.5)) / 2 / terms[0]
        
        if(t1 >= 1/2 and t1<= 1):
            return t1
        if(t2 >= 1/2 and t2<= 1):
            return t2
        
        return -1
    
    def solve_quadratic(self, p1, p2, p3):
        terms = self.compute_quadratic_coefficient(p1, p2, p3, self.p)
        t_temp = self.compute_t(terms)
        self.t = self.t + [t_temp]
        return t_temp
    
    def get_t(self):
        if(self.ft(self.vertices[0], self.vertices[1], self.vertices[2], self.p, 0.5)*self.ft(self.vertices[0], self.vertices[1], self.vertices[2], self.p, 1) <= 0):
            t_temp = self.solve_quadratic(self.vertices[0], self.vertices[1], self.vertices[2])
            self.P = self.P + [util.linear_interpolation(self.vertices[0], self.vertices[1], t_temp)]
            self.Q = self.Q + [util.linear_interpolation(self.vertices[0], self.vertices[2], 1/2/t_temp)]
            self.top = self.top + [self.vertices[0]]
            
        if(self.ft(self.vertices[1], self.vertices[2], self.vertices[0], self.p, 0.5)*self.ft(self.vertices[1], self.vertices[2], self.vertices[0], self.p, 1) <= 0):
            t_temp = self.solve_quadratic(self.vertices[1], self.vertices[2], self.vertices[0])
            self.P = self.P + [util.linear_interpolation(self.vertices[1], self.vertices[2], t_temp)]
            self.Q = self.Q + [util.linear_interpolation(self.vertices[1], self.vertices[0], 1/2/t_temp)]
            self.top = self.top + [self.vertices[1]]
            
        if(self.ft(self.vertices[2], self.vertices[0], self.vertices[1], self.p, 0.5)*self.ft(self.vertices[2], self.vertices[0], self.vertices[1], self.p, 1) <= 0):
            t_temp = self.solve_quadratic(self.vertices[2], self.vertices[0], self.vertices[1])
            self.P = self.P + [util.linear_interpolation(self.vertices[2], self.vertices[0], t_temp)]
            self.Q = self.Q + [util.linear_interpolation(self.vertices[2], self.vertices[1], 1/2/t_temp)]
            self.top = self.top + [self.vertices[2]]
            
    def v_decision_boundary(self):
        for i in range(len(self.P)):
            util.v_line_2D(self.P[i], self.Q[i])
            
    def v_triangle(self):
        util.v_triangle_2D(self.vertices[0], self.vertices[1], self.vertices[2])
        
    def v_vertices(self):
        plt.plot(self.vertices[0][0], self.vertices[0][1], 'o', color='r')
        plt.plot(self.vertices[1][0], self.vertices[1][1], 'o', color='r')
        plt.plot(self.vertices[2][0], self.vertices[2][1], 'o', color='r')
        
    def v_result(self):
        self.v_vertices()
        self.v_triangle()
        util.v_triangle_barycentre(self.vertices[0], self.vertices[1], self.vertices[2])
        self.get_t()
        self.v_decision_boundary()
        
    def print_info(self):
        print("The coordinates of point P: ", self.P)
        print("The coordinates of point Q: ", self.Q)
        print("The area of triangle: ", util.triangle_area_2D(self.vertices[0], self.vertices[1], self.vertices[2]))
        print("The area of small half-area triangle: ", util.triangle_area_2D(self.P[0], self.Q[0], self.top[0]))
```


```python
test = Triangle_Area_Bisection_2D(np.array([A,B,C]),np.array([0,0]))
test.v_result()
test.print_info()
```

    delta:  68.0
    The coordinates of point P:  [array([1.21922359, 0.34232922])]
    The coordinates of point Q:  [array([2.        , 0.56155281])]
    The area of triangle:  2.0
    The area of small half-area triangle:  0.9999999999999999



    
![png](output_11_1.png)
    


<a name='1-4'></a>
### 1.4 \*Interactive Plot

Please check out the **./triangle_bisector_interactive.py** file for more detail.

The program allow users to
+ Sepecify the 3 vertices of triangle interactively
+ Sepecify the point $R$ interactively
+ It will automatically generate the decision boundary

<a name='5'></a>
## 5. Reference

<a name='ref-1'></a>
1. [Craizer, Marcos. "Envelopes of Bisection Lines of Polygons." arXiv preprint arXiv:2203.10559 (2022).](https://arxiv.org/abs/2203.10559)
