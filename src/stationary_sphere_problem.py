import numpy as np
from figure import Figure
import G_functions
import envir_paths
import os
import customalgebra as ca

sphere_file = "Sphere_30_50.dat"
path_to_sphere = os.path.join(envir_paths.FIGURES_PATH, sphere_file)
sphere = Figure(path_to_sphere)


N = sphere.total_frames_in_objects[0]
frames = sphere.frames[0]
colloc = sphere.collocations[0]
colloc_dist = sphere.collocation_distances[0]
squares = sphere.squares[0]

print(frames.shape)
print(colloc.shape)
print(colloc_dist.shape)
print(squares.shape)

K_reg = np.ones((N, N))
#K_reg[:N, N] = np.zeros(N)
f_reg = np.zeros(N)

def f_right(colloc, q = np.array([1.1, 0, 0])):
    return 1/(4 * np.pi) * 1/ca.L2(colloc - q)

print(f_reg)
print(f_reg.shape)
print(K_reg)
print(K_reg.shape)


for i in range(N):
    for j in range(N):
        if i == j:
            K_reg[i, j] = 0
        else:
            K_reg[i, j] = 1/(4*np.pi) * 1/colloc_dist[i, j] * squares[j]


f_reg[:N] = np.ones(N)

#
# for i in range(N):
#     f_reg[i] = f_right(colloc=colloc[i], q=np.array([1.8, 0, 0]))

print(K_reg)
print(f_reg)

phi_reg = np.linalg.solve(K_reg, f_reg)

print(phi_reg)

file = open("test.txt", "w")
file.write(str(N) + "\n")
for i in range(N):
    file.write(str(round(phi_reg[i], 12)) + "\n")
file.close()