import math
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
import numpy as np


def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

x = np.array([1/17, 1/50, 1/100, 1/430, 1/160])
y = np.array([4270, 1350, 550, 430, 160])
m, b = np.polyfit(x, y, 1)
print(m, b)
plt.plot(x, y, 'ro')
plt.plot(x, m*x + b)
plt.xlabel("1/ρ [1/(nΩ*m)]")
plt.ylabel("TCR [ppm °C" + get_super("-1") + "]")
plt.show()