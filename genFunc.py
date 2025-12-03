import random
import os
import math


with open("nums.txt", "w") as file:
    file.write("num1,num2,result\n")
    for i in range(3000):
        a = random.randint(1, 200)
        b = random.randint(1, 200)
        res = a**0.5 + (b/a**b)
        file.write(f"{a},{b},{res:.3f}\n")
os.replace("nums.txt", "nums.csv")