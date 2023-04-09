import math


def calibrate(p):
    return p / (p + (1 - p) / 0.1)


def reverse_sigmoid(p):
    return math.log(p / (1 - p))
