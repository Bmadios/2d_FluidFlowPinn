# -*- coding: utf-8 -*-
"""
PINN Python Code pinn.py

@author: Blaise Madiega
email : blaisemadiega@gmail.com
"""

import torch
from torch.nn import Module, Linear, Tanh
from collections import OrderedDict
from utils import load_data

##CUDA Support
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  

class PINN_NavierStokes(Module):
    def __init__(self, layers):
        """
        Initialize the PINN model.
        
        Args:
        layers (list): A list specifying the number of units in each layer.
        """
        super(PINN_NavierStokes, self).__init__()

        self.depth = len(layers) - 1 
        self.activation = Tanh() 

        layer_list = []
        for i in range(self.depth-1):
            layer_list.append(('layer_%d' % i, Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation))
        layer_list.append(('layer_%d' % (self.depth-1), Linear(layers[-2], layers[-1])))

        self.layers = torch.nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
        x (Tensor): The input tensor.
        
        Returns:
        Tensor: The output tensor.
        """
        return self.layers(x)

def pde_loss(u_net, XY, Re):
  #x = torch.tensor(XY[:,0], requires_grad = True).float().cpu()
  #y = torch.tensor(XY[:, 1], requires_grad = True).float().cpu()
  x = XY[:, 0].clone().detach().requires_grad_(True).float().cpu()
  y = XY[:, 1].clone().detach().requires_grad_(True).float().cpu()
  xy_pde = torch.hstack((torch.reshape(x, (-1, 1)), torch.reshape(y, (-1, 1)))).float().to(device)

  u = u_net(xy_pde)[:, 0].to(device) # Velocity u
  v = u_net(xy_pde)[:, 1].to(device) # Velocity v
  p = u_net(xy_pde)[:, 2].to(device) # Pressure p

  x.to(device)
  y.to(device)

  u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True , create_graph=True)[0].to(device)
  u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True , create_graph=True)[0].to(device)
  u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True , create_graph=True)[0].to(device)
  u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True , create_graph=True)[0].to(device)

  v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True , create_graph=True)[0].to(device)
  v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True , create_graph=True)[0].to(device)
  v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True , create_graph=True)[0].to(device)
  v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True , create_graph=True)[0].to(device)

  p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True , create_graph=True)[0].to(device)
  p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True , create_graph=True)[0].to(device)

  # Loss computing
  f1 = u_x + v_y # Mass conservation
  f2 = u_xx/(Re) + u_yy/(Re) - p_x - u*u_x - v*u_y # Momentum eqn
  f3 = v_xx/(Re) + v_yy/(Re) - p_y - u*v_x - v*v_y

  loss1 = torch.mean(f1**2)
  loss2 = torch.mean(f2**2)
  loss3 = torch.mean(f3**2)

  total_loss = loss1 + loss2 + loss3
  return total_loss

def boundaries_loss(u_net, XY_INLET, XY_OUTLET, XY_WBOTTOM, XY_WTOP, XY_CIRC, U_inf):
  #x = torch.tensor(XY[:,0], requires_grad = True).float().cpu()
  #y = torch.tensor(XY[:, 1], requires_grad = True).float().cpu()
  x_in = XY_INLET[:, 0].clone().detach().requires_grad_(True).float().cpu()
  y_in = XY_INLET[:, 1].clone().detach().requires_grad_(True).float().cpu()
  xy_in = torch.hstack((torch.reshape(x_in, (-1, 1)), torch.reshape(y_in, (-1, 1)))).float().to(device)

  x_out = XY_OUTLET[:, 0].clone().detach().requires_grad_(True).float().cpu()
  y_out = XY_OUTLET[:, 1].clone().detach().requires_grad_(True).float().cpu()
  xy_out = torch.hstack((torch.reshape(x_out, (-1, 1)), torch.reshape(y_out, (-1, 1)))).float().to(device)

  x_bot = XY_WBOTTOM[:, 0].clone().detach().requires_grad_(True).float().cpu()
  y_bot = XY_WBOTTOM[:, 1].clone().detach().requires_grad_(True).float().cpu()
  xy_bot = torch.hstack((torch.reshape(x_bot, (-1, 1)), torch.reshape(y_bot, (-1, 1)))).float().to(device)

  x_top = XY_WTOP[:, 0].clone().detach().requires_grad_(True).float().cpu()
  y_top = XY_WTOP[:, 1].clone().detach().requires_grad_(True).float().cpu()
  xy_top = torch.hstack((torch.reshape(x_top, (-1, 1)), torch.reshape(y_top, (-1, 1)))).float().to(device)

  x_c = XY_CIRC[:, 0].clone().detach().requires_grad_(True).float().cpu()
  y_c = XY_CIRC[:, 1].clone().detach().requires_grad_(True).float().cpu()
  xy_c = torch.hstack((torch.reshape(x_c, (-1, 1)), torch.reshape(y_c, (-1, 1)))).float().to(device)

  u_in = u_net(xy_in)[:, 0].to(device) # Velocity u_in
  v_in = u_net(xy_in)[:, 1].to(device) # Velocity v_in
  p_in = u_net(xy_in)[:, 2].to(device) # Pressure p_in


  u_out = u_net(xy_out)[:, 0].to(device) # Velocity u_out
  v_out = u_net(xy_out)[:, 1].to(device) # Velocity v_out
  p_out = u_net(xy_out)[:, 2].to(device) # Pressure p_out

  u_bot = u_net(xy_bot)[:, 0].to(device) # Velocity u_wall bottom
  v_bot = u_net(xy_bot)[:, 1].to(device) # Velocity v_wall bottom
  p_bot = u_net(xy_bot)[:, 2].to(device) # Pressure p_wall bottom

  u_top = u_net(xy_top)[:, 0].to(device) # Velocity u_wall top
  v_top = u_net(xy_top)[:, 1].to(device) # Velocity v_wall top
  p_top = u_net(xy_top)[:, 2].to(device) # Pressure p_wall top

  u_c = u_net(xy_c)[:, 0].to(device) # Velocity u_circle
  v_c = u_net(xy_c)[:, 1].to(device) # Velocity v_circle
  p_c = u_net(xy_c)[:, 2].to(device) # Pressure p_circle

  x_in.to(device)
  y_in.to(device)
  x_out.to(device)
  y_out.to(device)

  x_bot.to(device)
  y_bot.to(device)
  x_top.to(device)
  y_top.to(device)

  x_c.to(device)
  y_c.to(device)

  ##### GRADIENTS INLET ###########
  u_x_in = torch.autograd.grad(u_in, x_in, grad_outputs=torch.ones_like(u_in), retain_graph=True , create_graph=True)[0].to(device)
  u_y_in = torch.autograd.grad(u_in, y_in, grad_outputs=torch.ones_like(u_in), retain_graph=True , create_graph=True)[0].to(device)
  

  v_x_in = torch.autograd.grad(v_in, x_in, grad_outputs=torch.ones_like(v_in), retain_graph=True , create_graph=True)[0].to(device)
  v_y_in = torch.autograd.grad(v_in, y_in, grad_outputs=torch.ones_like(v_in), retain_graph=True , create_graph=True)[0].to(device)

  p_x_in = torch.autograd.grad(p_in, x_in, grad_outputs=torch.ones_like(p_in), retain_graph=True , create_graph=True)[0].to(device)
  p_y_in = torch.autograd.grad(p_in, y_in, grad_outputs=torch.ones_like(p_in), retain_graph=True , create_graph=True)[0].to(device)

  ##### GRADIENTS OUTLET ###########
  u_x_out = torch.autograd.grad(u_out, x_out, grad_outputs=torch.ones_like(u_out), retain_graph=True , create_graph=True)[0].to(device)
  u_y_out = torch.autograd.grad(u_out, y_out, grad_outputs=torch.ones_like(u_out), retain_graph=True , create_graph=True)[0].to(device)
  

  v_x_out = torch.autograd.grad(v_out, x_out, grad_outputs=torch.ones_like(v_out), retain_graph=True , create_graph=True)[0].to(device)
  v_y_out = torch.autograd.grad(v_out, y_out, grad_outputs=torch.ones_like(v_out), retain_graph=True , create_graph=True)[0].to(device)

  p_x_out = torch.autograd.grad(p_out, x_out, grad_outputs=torch.ones_like(p_out), retain_graph=True , create_graph=True)[0].to(device)
  p_y_out = torch.autograd.grad(p_out, y_out, grad_outputs=torch.ones_like(p_out), retain_graph=True , create_graph=True)[0].to(device)
    
  ##### U GRADIENTS BOTTOM WALL ###########
  u_x_bot = torch.autograd.grad(u_bot, x_bot, grad_outputs=torch.ones_like(u_bot), retain_graph=True , create_graph=True)[0].to(device)
  u_y_bot = torch.autograd.grad(u_bot, y_bot, grad_outputs=torch.ones_like(u_bot), retain_graph=True , create_graph=True)[0].to(device)
  

  v_x_bot = torch.autograd.grad(v_bot, x_bot, grad_outputs=torch.ones_like(v_bot), retain_graph=True , create_graph=True)[0].to(device)
  v_y_bot = torch.autograd.grad(v_bot, y_bot, grad_outputs=torch.ones_like(v_bot), retain_graph=True , create_graph=True)[0].to(device)
    
  ##### U GRADIENTS TOP WALL ###########
  u_x_top = torch.autograd.grad(u_top, x_top, grad_outputs=torch.ones_like(u_top), retain_graph=True , create_graph=True)[0].to(device)
  u_y_top = torch.autograd.grad(u_top, y_top, grad_outputs=torch.ones_like(u_top), retain_graph=True , create_graph=True)[0].to(device)
  

  v_x_top = torch.autograd.grad(v_top, x_top, grad_outputs=torch.ones_like(v_top), retain_graph=True , create_graph=True)[0].to(device)
  v_y_top = torch.autograd.grad(v_top, y_top, grad_outputs=torch.ones_like(v_top), retain_graph=True , create_graph=True)[0].to(device)

  ##### P GRADIENTS WALL BOTTOM ###########
  p_x_bot = torch.autograd.grad(p_bot, x_bot, grad_outputs=torch.ones_like(p_bot), retain_graph=True , create_graph=True)[0].to(device)
  p_y_bot = torch.autograd.grad(p_bot, y_bot, grad_outputs=torch.ones_like(p_bot), retain_graph=True , create_graph=True)[0].to(device)

  ##### GRADIENTS WALL TOP ###########
  p_x_top = torch.autograd.grad(p_top, x_top, grad_outputs=torch.ones_like(p_top), retain_graph=True , create_graph=True)[0].to(device)
  p_y_top = torch.autograd.grad(p_top, y_top, grad_outputs=torch.ones_like(p_top), retain_graph=True , create_graph=True)[0].to(device)

  ##### GRADIENTS obstacle: CYLINDER ###########
  p_x_c = torch.autograd.grad(p_c, x_c, grad_outputs=torch.ones_like(p_c), retain_graph=True , create_graph=True)[0].to(device)
  p_y_c = torch.autograd.grad(p_c, y_c, grad_outputs=torch.ones_like(p_c), retain_graph=True , create_graph=True)[0].to(device)




  # Loss computing
  fbc_inlet_u = u_in - U_inf # inlet u
  fbc_inlet_v = v_in # inlet v
  fbc_inlet_p1 = p_x_in # inlet p 
  fbc_inlet_p2 = p_y_in # inlet p
  ###
  fbc_inlet_p = p_x_in + p_y_in # inlet p
  ###

  fbc_outlet_u1 = u_x_out  # outlet u gradient zero x
  fbc_outlet_u2 = u_y_out # outlet u gradient zero y

  fbc_outlet_u = u_x_out + u_y_out   # outlet u gradient zero x

  fbc_outlet_v1 = v_x_out  # outlet v gradient zero x
  fbc_outlet_v2 = v_y_out # outlet v gradient zero y
    
  fbc_outlet_v = v_x_out + v_y_out
    
  fbc_outlet_p = p_out # outlet p = 0

  fbc_bottom_u = u_x_bot  # ux bottom = 0
  fbc_bottom_v = v_bot # v bottom = 0
  fbc_bottom_p1 = p_x_bot  # Zerogradient Bottom
  fbc_bottom_p2 = p_y_bot # Zerogradient Bottom

  fbc_bottom_p = p_x_bot + p_y_bot   # Zerogradient Bottom

  fbc_top_u = u_x_top # ux top = 0
  fbc_top_v = v_top # v top = 0
  fbc_top_p1 = p_x_top  # Zerogradient top
  fbc_top_p2 = p_y_top # Zerogradient top
    
  fbc_top_p = p_x_top + p_y_top  # Zerogradient top

  fbc_c_u = u_c # u circle = 0
  fbc_c_v = v_c # v circle = 0
  fbc_c_p1 = p_x_c # Zerogradient inside circle
  fbc_c_p2 = p_y_c # Zerogradient inside circle

  fbc_c_p = p_x_c +  p_y_c # Zerogradient inside circle


  #loss_BC_inlet = torch.mean(fbc_inlet_u**2) + torch.mean(fbc_inlet_v**2) + torch.mean(fbc_inlet_p1**2) + torch.mean(fbc_inlet_p2**2)
  loss_BC_inlet = torch.mean(fbc_inlet_u**2) + torch.mean(fbc_inlet_v**2) # + torch.mean(fbc_inlet_p**2)
  #loss_BC_outlet = torch.mean(fbc_outlet_u1**2) + torch.mean(fbc_outlet_u2**2) + torch.mean(fbc_outlet_v1**2) + torch.mean(fbc_outlet_v2**2) +    torch.mean(fbc_outlet_p**2)
  loss_BC_outlet = torch.mean(fbc_outlet_u**2)  + torch.mean(fbc_outlet_v**2)  # + torch.mean(fbc_outlet_p**2)
  #loss_BC_bottom = torch.mean(fbc_bottom_u**2) + torch.mean(fbc_bottom_v**2) + torch.mean(fbc_bottom_p1**2) + torch.mean(fbc_bottom_p2**2)
  loss_BC_bottom = torch.mean(fbc_bottom_u**2) + torch.mean(fbc_bottom_v**2) # + torch.mean(fbc_bottom_p**2)
  #loss_BC_top = torch.mean(fbc_top_u**2) + torch.mean(fbc_top_v**2) + torch.mean(fbc_top_p1**2) + torch.mean(fbc_top_p2**2)
  loss_BC_top = torch.mean(fbc_top_u**2) + torch.mean(fbc_top_v**2) # + torch.mean(fbc_top_p**2)
  #loss_BC_c = torch.mean(fbc_c_u**2) + torch.mean(fbc_c_v**2) + torch.mean(fbc_c_p1**2) + torch.mean(fbc_c_p2**2)
  loss_BC_c = torch.mean(fbc_c_u**2) + torch.mean(fbc_c_v**2) # + torch.mean(fbc_c_p**2)

  total_loss = loss_BC_inlet + loss_BC_outlet + loss_BC_bottom + loss_BC_top + loss_BC_c
  return total_loss


# LOSS FUNCTIONS
def loss_func(u_net, XY_pde, Re, U_inf, XY_IN, XY_OUT, XY_BOT, XY_TOP, XY_CIRC):
  """
  Computes the total loss for the problem. The total loss is a sum of the loss of interior points of PDE
  and the boundary loss evaluation.
  
  Args:
    u_net (nn.Module): The neural network model.
    XY_pde (torch.tensor): The collocation points for the PDE.
    Re (float): The Reynolds number.
    U_inf (float): The free stream velocity.
    XY_IN (torch.tensor): The collocation points for the inlet.
    XY_OUT (torch.tensor): The collocation points for the outlet.
    XY_BOT (torch.tensor): The collocation points for the bottom wall.
    XY_TOP (torch.tensor): The collocation points for the top wall.
    XY_CIRC (torch.tensor): The collocation points for the boundary of the circle.
    
  Returns:
    torch.tensor: The total loss.
  """
  loss_pde = pde_loss(u_net, XY_pde, Re) #loss interior points of PDE
  loss_bounds = boundaries_loss(u_net, XY_IN, XY_OUT, XY_BOT, XY_TOP, XY_CIRC, U_inf) # boundaries loss evaluation
  loss_total_PDE = loss_pde + loss_bounds
  return loss_total_PDE

def hard_loss_func(u_net, w_pde, XY_data, UVP_data, loss_pde):
    """
    Computes the total hard loss. The hard loss is a sum of the weight of PDE loss and the mean squared error
    between the predicted and actual values.
  
    Args:
      u_net (nn.Module): The neural network model.
      w_pde (float): The weight of the PDE loss.
      XY_data (torch.tensor): The collocation points for the data.
      UVP_data (torch.tensor): The actual values at the collocation points.
      loss_pde (torch.tensor): The PDE loss.
  
    Returns:
      torch.tensor: The total hard loss.
    """
    res_data = u_net(XY_data) - UVP_data
    total_loss = w_pde*loss_pde + torch.mean(res_data**2)
    return total_loss