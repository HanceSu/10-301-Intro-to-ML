import math
x = 1.0
a, b, c = (0.0, 0.0, 0.0)
for i in range(10000):
    new_c = 0.1 * x + 0.8 * c + 0.1 * b
    new_b = 0.1 * c + 0.8 * b + 0.1 * a
    new_a = 0.1 * b + 0.8 * a + 0.1 * a
    a, b, c = new_a, new_b, new_c

print(a, b, c)