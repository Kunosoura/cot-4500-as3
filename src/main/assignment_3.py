import numpy as np

# function definitions

def Euler_Method(function, minVal, maxVal, iterVal, initVal):
  t = minVal
  y = initVal
  dt = (maxVal - minVal) / iterVal

  for i in range(iterVal):
    dy = eval(function)
    t += dt
    y += dy * dt
  
  return(y)

def Runge_Kutta(function, minVal, maxVal, iterVal, initVal):
  # subfunction that returns differential at the given point
  def Diffy_Q(t,y):
    return(eval(function))
  
  t = minVal
  y = initVal
  dt = (maxVal - minVal) / iterVal

  for i in range(iterVal):
    k1 = dt * Diffy_Q(t,y)
    k2 = dt * Diffy_Q(t + dt/2, y + k1/2)
    k3 = dt * Diffy_Q(t + dt/2, y + k2/2)
    k4 = dt * Diffy_Q(t + dt, y + k3)
    t += dt
    y += (k1 + 2*k2 + 2*k3 + k4)/6
  
  return(y)

def Gaussian_Elimination(augMat):
  # method assures that all elements remain ints if the matrix begins with ints
  for i in range(0, len(augMat) - 1):
    # zeroing column i, iterating down from diagonal element
    for j in range(i + 1, len(augMat)):
      augMat[j,:] = augMat[j,:] * augMat[i,i] - augMat[j,i] * augMat[i,:]

  # backward substitution, starting at penultimate row, going backwards
  for i in range(len(augMat) - 2, -1, -1):
    for j in range(len(augMat) - 1, i, -1):
      augMat[i,:] = augMat[i,:] * augMat[j,j]- augMat[i,j] * augMat[j,:]
  
  # solution is last element in each row
  # divided by corresponding diagonal element
  return(augMat[:,len(augMat)]/np.diagonal(augMat)[:])

def Det_L_U(matrixU):
  # method assures that all elements remain ints if the matrix begins with ints
  # solution key asks for floats, so setting type to float anyway
  matrixL = np.identity(len(matrixU))
  matrixU = matrixU.astype(float)

  print("%.5f" % np.linalg.det(matrixU), end = "\n\n")

  # check if algorithm is applicable
  for i in range(1, len(matrixU)+1):
    if np.linalg.det(matrixU[0:i,0:i]) == 0:
      return ("Error: not all leading principle submatrices are nonsingular")

  for i in range(0, len(matrixU) - 1):
    # zeroing column i, iterating down from diagonal element
    # filling L matrix with scalars that are used on U matrix rows
    for j in range(i + 1, len(matrixU)):
      if matrixU[j,i] != 0: # skip if already 0
        matrixL[j,i] = matrixU[j,i] * np.sign(matrixU[i,i])
        matrixU[j,:] = matrixU[j,:] * abs(matrixU[i,i]) - np.sign(matrixU[i,i]) * matrixU[j,i] * matrixU[i,:]

  print(matrixL, end = "\n\n")
  print(matrixU, end = "\n\n")

def Diag_Dom(matrix):
  for i in range(len(matrix)):
    # check each pivot in each row
    for j in range(len(matrix)):
      # check if any other elements in row are larger than pivot
      if i!= j:
        if matrix[i,j] > matrix[i,i]:
          # conidition met, no need to keep looking
          return("True")
  
  # otherwise, matrix is not diagonally dominate
  return("False")

def Pos_Def(matrix):
  # returns false if matrix is not square or not symmetric
  if not np.array_equal(matrix.transpose(), matrix):
    return("False")

  # returns false if eigenvalues aren't positive
  if min(np.linalg.eigvals(matrix))<=0:
    return("False")
  
  # otherwise, matrix is positive definite
  return("True")

# =============== BEGINNING OF "MAIN" =================

# not sure if this is needed for this assignment, but anway
np.set_printoptions(precision=7, suppress=True, linewidth=100)

print("%.5f" % Euler_Method("t - y**2", 0, 2, 10, 1), end = "\n\n")

print("%.5f" % Runge_Kutta("t - y**2", 0, 2, 10, 1), end = "\n\n")

print(Gaussian_Elimination(np.array(
    [[ 2,-1, 1, 6],
     [ 1, 3, 1, 0],
     [-1, 5, 4,-3]])), end = "\n\n")

Det_L_U(np.array(
    [[ 1, 1, 0, 3],
     [ 2, 1,-1, 1],
     [ 3,-1,-1, 2],
     [-1, 2, 3,-1]]))

print(Diag_Dom(np.array(
    [[ 9, 0, 5, 2, 1],
     [ 3, 9, 1, 2, 1],
     [ 0, 1, 7, 2, 3],
     [ 4, 2, 3,12, 2],
     [ 3, 2, 4, 0, 8]])), end = "\n\n")

print(Pos_Def(np.array(
    [[2,2,1],
     [2,3,0],
     [1,0,2]])))