# -*- coding: utf-8 -*-
"""
Code file utils.py 

@author: Blaise Madiega
email : blaisemadiega@gmail.com
"""

import math
import torch
import random
import numpy as np
import pandas as pd

##CUDA Support
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

def exp_decay_schedule(epoch, initial_lr, decay_rate, plateau_epoch):
    """
    Calculates the exponentially decaying learning rate for each epoch.

    Args:
        epoch (int): The current training epoch.
        initial_lr (float): The initial learning rate.
        decay_rate (float): The decay rate for the learning rate.
        plateau_epoch (int): The epoch after which learning rate decay begins.

    Returns:
        float: The learning rate for the current epoch.
    """
    lr = initial_lr * math.exp(-decay_rate * (epoch / plateau_epoch))
    return max(lr, 1e-5)

def init_weights(m):
    """
    Initializes the weights and biases of the provided model.

    Args:
        m (torch.nn.Module): The model whose weights and biases should be initialized.

    Returns:
        None
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)

def colloc_pde_interior(N_x, N_y, x_min, x_max, y_min, y_max, x_center, y_center, r_cyl):
    """
    Generates random interior collocation points and returns those that are outside a specified boundary.

    Args:
        N_x (int): Number of points in x-direction.
        N_y (int): Number of points in y-direction.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate.
        y_min (float): Minimum y-coordinate.
        y_max (float): Maximum y-coordinate.
        x_center (float): x-coordinate of the center of the cylinder.
        y_center (float): y-coordinate of the center of the cylinder.
        r_cyl (float): Radius of the cylinder.

    Returns:
        tensor: Randomly generated interior collocation points that are outside the specified boundary.
    """
    xy = []
    for i in range(N_x*N_y):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        xy.append([x, y])
    indices = []
    for i in range(len(xy)):
        if xy[i][1] == y_min or xy[i][1] == y_max or xy[i][0] == x_min or xy[i][0] == x_max or ((xy[i][0] - x_center)**2 + (xy[i][1] - y_center)**2) <= r_cyl**2:
            indices.append(i)
    xy = np.delete(xy, indices, axis = 0)
    xy = torch.from_numpy(xy).float()
    return xy

def colloc_pde_with_mesh(N_x, N_y, x_min, x_max, y_min, y_max, x_center, y_center, r_cyl):
  """
    Generates random interior collocation points using MESHING of domain strategy.

    Args:
        N_x (int): Number of points in x-direction.
        N_y (int): Number of points in y-direction.
        x_min (float): Minimum x-coordinate.
        x_max (float): Maximum x-coordinate.
        y_min (float): Minimum y-coordinate.
        y_max (float): Maximum y-coordinate.
        x_center (float): x-coordinate of the center of the cylinder.
        y_center (float): y-coordinate of the center of the cylinder.
        r_cyl (float): Radius of the cylinder.

    Returns:
        tensor: Randomly generated interior collocation points that are outside the specified boundary.
    """
  xy = []
  for i in range(N_x*N_y):
    x_1 = random.uniform(x_min, 10.0)
    y_1 = random.uniform(0.0, 4.5) 
    x_2 = random.uniform(10.0, 20.0)
    y_2 = random.uniform(0.0, 4.5) 
    x_3 = random.uniform(20.0, 28.0)
    y_3 = random.uniform(0.0, 4.5) 
    x_4 = random.uniform(20.0, 28.0)
    y_4 = random.uniform(4.5, 9.0) 
    x_5 = random.uniform(10.0, 20.0)
    y_5 = random.uniform(4.5, 9.0) 
    x_6 = random.uniform(0.0, 10.0)
    y_6 = random.uniform(4.5, 9.0) 
    xy.append([x_1, y_1])
    xy.append([x_2, y_2])
    xy.append([x_3, y_3])
    xy.append([x_4, y_4])
    xy.append([x_5, y_5])
    xy.append([x_6, y_6])
  
  indices = []
  x_inlet = x_min
  x_outlet = x_max
  y_wall_bottom = y_min
  y_wall_top = y_max
  #compteur = 0
  for i in range(len(xy)):
    if xy[i][1] == y_wall_bottom or xy[i][1] == y_wall_top or xy[i][0] == x_inlet or xy[i][0] == x_outlet or ((xy[i][0] - x_center)**2 + (xy[i][1] - y_center)**2) <= r_cyl**2:

      indices.append(i)

  xy = np.delete(xy, indices, axis = 0)
  xy = torch.from_numpy(xy).float().to(device)
  return xy

def colloc_Xinlet(Nin, x_min, y_min, y_max):
  """
  Generates the inlet collocation points.
  
  Args:
    Nin (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    y_min (float): The minimum y coordinate of the collocation points.
    y_max (float): The maximum y coordinate of the collocation points.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  xy = []
  for i in range(Nin*Nin):
    x = x_min
    y = random.uniform(y_min, y_max)
    xy.append([x, y])
  
  xy = np.array(xy)
  xy = torch.from_numpy(xy).float().to(device)
  return xy

def colloc_Xoutlet(Nout, x_max, y_min, y_max):
  """
  Generates the outlet collocation points.
  
  Args:
    Nout (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    y_min (float): The minimum y coordinate of the collocation points.
    y_max (float): The maximum y coordinate of the collocation points.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  xy = []
  for i in range(Nout*Nout):
    x = x_max
    y = random.uniform(y_min, y_max)
    xy.append([x, y])
  
  xy = np.array(xy)
  xy = torch.from_numpy(xy).float().to(device)
  return xy

def colloc_Wall_bottom(Nwallbot, x_min, x_max, y_min):
  """
  Generates the bottom wall collocation points.
  
  Args:
    Nwallbot (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    x_max (float): The maximum x coordinate of the collocation points.
    y_min (float): The y coordinate of the bottom wall.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  xy = []
  for i in range(Nwallbot*Nwallbot):
    x = random.uniform(x_min, x_max)
    y = y_min
    xy.append([x, y])
  
  xy = np.array(xy)
  xy = torch.from_numpy(xy).float().to(device)
  return xy

def colloc_Wall_top(Nwalltop, x_min, x_max, y_max):
  """
  Generates the top wall collocation points.
  
  Args:
    Nwalltop (int): The square root of the total number of collocation points.
    x_min (float): The minimum x coordinate of the collocation points.
    x_max (float): The maximum x coordinate of the collocation points.
    y_min (float): The y coordinate of the bottom wall.
  
  Returns:
    torch.tensor: The collocation points as a torch tensor.
  """
  xy = []
  for i in range(Nwalltop*Nwalltop):
    x = random.uniform(x_min, x_max)
    y = y_max
    xy.append([x, y])
  
  xy = np.array(xy)
  xy = torch.from_numpy(xy).float().to(device)
  return xy

def colloc_BC_circle(N, x_center, y_center, r_cyl):
    """
    Generates the boundary collocation points on the circle.
    
    Args:
      N (int): The square root of the total number of collocation points.
      x_center (float): The x coordinate of the center of the circle.
      y_center (float): The y coordinate of the center of the circle.
      r_cyl (float): The radius of the circle.
    
    Returns:
      torch.tensor: The collocation points as a torch tensor.
    """
    xy = []

    for i in range(N*N):
        while True:
            x = np.random.uniform(x_center - r_cyl, x_center + r_cyl)
            y = np.random.uniform(y_center - r_cyl, y_center + r_cyl)
            if ((x - x_center) ** 2 + (y - y_center) ** 2) <= r_cyl ** 2:
                xy.append([x, y])
                break

    xy = np.array(xy)  # Convert the list to a NumPy array
    xy = torch.from_numpy(xy).float().to(device)
    return xy

def load_data(file_path, N_mes, device):
    """Loads and preprocesses data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        N_mes (int): The number of random samples to take from the data.
        device (str): The device to send the tensors to.

    Returns:
        Tuple: Tuple containing:
            random_XY_data (torch.Tensor): Random samples of XY data.
            XY_data (torch.Tensor): All XY data.
            random_UVP_data (torch.Tensor): Random samples of UVP data.
            UVP_data (torch.Tensor): All UVP data.
            BC_XY_data (torch.Tensor): Filtered XY data.
            BC_UVP_data (torch.Tensor): Filtered UVP data.
    """
    data = pd.read_csv(file_path)

    X_dat = data["Points:0"].values + 10
    Y_dat = data["Points:1"].values + 4.5
    u_dat = data["U:0"].values
    v_dat = data["U:1"].values
    U_dat = np.sqrt(u_dat**2 + v_dat**2)
    p_dat = data["p"].values

    xy = np.array([[x, y] for x, y in zip(X_dat, Y_dat)])
    uvp_data = np.array([[u, v, p] for u, v, p in zip(u_dat, v_dat, p_dat)])

    random_indices = random.sample(range(len(xy)), N_mes)
    random_XY_data = torch.from_numpy(xy[random_indices]).float().to(device)
    random_UVP_data = torch.from_numpy(uvp_data[random_indices]).float().to(device)
    XY_data = torch.from_numpy(xy).float().to(device)
    UVP_data = torch.from_numpy(uvp_data).float().to(device)

    BC_indices = [i for i, (x, y) in enumerate(xy) if x in {0, 28} or y in {0, 9} or ((x-10)**2 + (y-4.5)**2 <= 0.5**2)]
    BC_XY_data = torch.from_numpy(xy[BC_indices]).float().to(device)
    BC_UVP_data = torch.from_numpy(uvp_data[BC_indices]).float().to(device)

    return random_XY_data, XY_data, random_UVP_data, UVP_data, BC_XY_data, BC_UVP_data