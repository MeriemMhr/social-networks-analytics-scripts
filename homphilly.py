# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:22:44 2024
"""
import numpy as np

# Function to create a matrix with specified rows and columns filled with zeros
def create_matrix(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]

# Function to fill the matrix with specified numbers
def fill_matrix(matrix, numbers):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = numbers[i][j]

# Create two 7x7 matrices

matrix1 = create_matrix(7, 7)
#social ties matrix


matrix2 = create_matrix(7, 7)
#attribute ties matrix

# Define different numbers to fill the matrices
numbers1 = [
    [0, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0]
]

numbers2 = [
   [0, 0, 0, 1, 0, 1, 1],
   [0, 0, 1, 0, 1, 0, 0],
   [0, 1, 0, 0, 1, 0, 0],
   [1, 0, 0, 0, 0, 1, 1],
   [0, 1, 1, 0, 0, 0, 0],
   [1, 0, 0, 1, 0, 0, 1],
   [1, 0, 0, 1, 0, 1, 0]
]

# Fill the matrices with the specified numbers
fill_matrix(matrix1, numbers1)
fill_matrix(matrix2, numbers2)

# Print the matrices (optional)
print("Matrix 1:")
for row in matrix1:
    print(row)

print("\nMatrix 2:")
for row in matrix2:
    print(row)

matrix3= np.triu(matrix2,1)
# upper triangular conversion of attribute tie

print("\nMatrix 3:")
for row in matrix3:
    print(row)
    
print("\nsame value:")
print(matrix3.sum())

matrix_samenet_samevale= np.dot(matrix1, matrix2)

print("\nMatrix same net same val:")
for row in matrix_samenet_samevale:
    print(row)

print("\na:")
a = np.trace(matrix_samenet_samevale)/2;
print(np.trace(matrix_samenet_samevale)/2)

c = matrix3.sum()-a

mat_size=  21

notsame_val= mat_size - matrix3.sum()


matrix4= np.triu(matrix1,1)
#upper triangukar conversion of social ties matrix

print("\nMatrix 4:")
for row in matrix4:
    print(row)
    
print("\nsame_net:")
print(matrix4.sum())

b = matrix4.sum()-a

d = notsame_val - b

print (a , b, c, d)


print("\ncoeff:")
print (pow((a*d-c*b),2)*21 / (8*13*9*12))