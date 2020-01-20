# -*- coding: utf-8 -*-
# use GD to bring X2 and X4
import numpy as np
# square of x
def x2(x):
    return (x*x)

# derivative of x2
def x2_(x):
    return 2*x

"""## Ex2_1 (a) find the minimums of x^2 using Gradient Descent & Mumentum"""

# starting point 
X2 = 10
X2m = 10
mu = 0.5
v = 0

lr = 0.001 #your code
lrm = 0.001 #your code
num_of_steps = 10000


for i in range(num_of_steps):
    gradient = - x2_(X2)
    gradient_m = - x2_(X2m)
    v = mu * v + lrm * gradient_m
    X2 = X2 + gradient * lr
    X2m = X2m + v
    if i%100 == 0:
        print("Step:{} \t X2:{} \t X2m:{} ".format(i, X2, X2m))


# =============================================================================
# Examples for print formatting:
#  
# if i%10000 == 0:
#        print("Step:{} \t X2:{} \t X2m:{} ".format(i,X2, X2m))
#        
# =============================================================================
  
  
"""## Ex2_1 (b) find the minimums of x^4 using Gradient Descent & Mumentum"""

# x to the power of 4
def x4(x):
     return x**4
 
 # derivative of x4
def x4_(x):
     return 4*x**3


# starting point 
X4 = 10
X4m = 10
mu = 0.5
v = 0

lr = 0.001 #your code
lrm = 0.001 #your code
num_of_steps = 100000

for i in range(num_of_steps):
    gradient = - x4_(X4)
    gradient_m = - x4_(X4m)
    v = mu * v + lrm * gradient_m
    X4 = X4 + gradient * lr
    X4m = X4m + v
    if i % 1000 == 0:
        print("Step:{} \t X4:{} \t X4m:{} ".format(i, X4, X4m))
  
# =============================================================================
# Examples for print formatting:
#  
# if i%10000 == 0:
#        print("Step:{} \t X2:{} \t X2m:{} ".format(i,X2, X2m))
#        
# =============================================================================
