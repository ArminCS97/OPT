import numpy as np
from scipy.optimize import minimize
from math import e


# 1: x[0] is the same as x1 and x[1] is the same as x2


def penalty_method():
    c = 0
    for i in range(30):
        def q(x):  # the function q(x1, x2, c)
            return -(x[0] + 1) * x[1] + c * (x[0] + x[1] - 3) ** 2

        answer = minimize(q, x0=np.array([0, 0]))  # using minimize function of scipy.optimize
        print(f"The value of min for c = {c} is ", answer.x)
        c += 100


penalty_method()
# Below you can see the output of penalty_method()
"""
The value of min for c = 0 is  [7.22986163e+08 2.58766826e+17]
The value of min for c = 100 is  [1.00501111 2.00501396]
The value of min for c = 200 is  [1.00250312 2.00250313]
The value of min for c = 300 is  [1.00166809 2.00166802]
The value of min for c = 400 is  [1.0012508  2.00125077]
The value of min for c = 500 is  [1.00100048 2.00100052]
The value of min for c = 600 is  [1.00083368 2.00083368]
The value of min for c = 700 is  [1.00071225 2.00071683]
The value of min for c = 800 is  [1.0006252  2.00062518]
The value of min for c = 900 is  [1.00055568 2.00055573]
The value of min for c = 1000 is  [1.00050013 2.00050011]
The value of min for c = 1100 is  [1.00045467 2.00045462]
The value of min for c = 1200 is  [1.00041675 2.00041675]
The value of min for c = 1300 is  [1.00038468 2.00038469]
The value of min for c = 1400 is  [1.00036407 2.00035034]
The value of min for c = 1500 is  [1.00033337 2.0003334 ]
The value of min for c = 1600 is  [1.00031254 2.00031255]
The value of min for c = 1700 is  [1.00029414 2.00029417]
The value of min for c = 1800 is  [1.0002778  2.00027782]
The value of min for c = 1900 is  [1.00026317 2.00026321]
The value of min for c = 2000 is  [1.00025002 2.00025004]
The value of min for c = 2100 is  [1.00023811 2.00023813]
The value of min for c = 2200 is  [1.0002273  2.00022729]
The value of min for c = 2300 is  [1.00021742 2.00021741]
The value of min for c = 2400 is  [1.00020836 2.00020835]
The value of min for c = 2500 is  [1.00020002 2.00020001]
The value of min for c = 2600 is  [1.00019231 2.00019233]
The value of min for c = 2700 is  [1.00018521 2.00018519]
The value of min for c = 2800 is  [1.00017861 2.00017856]
The value of min for c = 2900 is  [1.00017241 2.00017244]
"""


def barrier_method():
    for c in range(1, 25):
        def q(x):  # the function q(x1, x2, c)
            return e ** (x[0] ** 2 + x[1] ** 2) - 1 / c * (1 / (x[0] + x[1] + 4))

        def dqx0(x):  # dq/dx1 function
            return 2 * x[0] * e ** (x[0] ** 2 + x[1] ** 2) - 1 / c * (-1 / (x[0] + x[1] + 4) ** 2)

        def dqx1(x):  # dq/dx2 function
            return 2 * x[1] * e ** (x[0] ** 2 + x[1] ** 2) - 1 / c * (-1 / (x[0] + x[1] + 4) ** 2)

        x0 = np.array([0, 0])  # x0 = [0, 0] as the initial point
        gradient_of_x0 = np.array([dqx0(x0), dqx1(x0)])  # the same as gradient_of_q(x1=0, x2=0)

        def steepest_descent(a):  # steepest descent function in terms of an a (the same as alpha)
            return q(x0 - a * gradient_of_x0)

        a0 = minimize(steepest_descent, x0=x0).x  # minimize the steepest descent function to
        # get the best a (the sane as alpha) this a will be used to estimate the next iteration
        next_iteration = x0 - a0 * gradient_of_x0  # next iteration according to steepest_descent(a)
        print(f"The value of min for c = {c} is ", next_iteration)


def q(x):  # the function q(x1, x2, c)
    return e ** (x[0] ** 2 + x[1] ** 2) - (1 / 10000000) * (1 / (x[0] + x[1] + 4))

b = minimize(q, x0=np.array([1, 1]))
print(b.x)
