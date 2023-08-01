# -*- coding: utf-8 -*-
"""
Code file main.py
This is the main Python file for performing flow reconstruction using PINNs
See details in the readme.md.

@author: Blaise Madiega
email : blaisemadiega@gmail.com
"""

import torch
import numpy as np
from argparse import ArgumentParser
from pinn import *
from utils import *
from plotting import *
import csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # device

def main(args):
    file_path = args.data_path
    N_mes = 1000  # number of measurements points
    
    x_min, x_max = 0.0, 28.0
    y_min, y_max = 0.0, 9.0
    x_center, y_center = 10, 4.5 # Center of cylinder
    r_cyl = 0.5 # rayon of cylinder
    W_pde = 1.0
    
    dat = pd.read_csv(file_path)
    timestep = dat["Time"].values
    X_dat = dat["Points:0"].values
    x_data = []
    y_data = []
    for i in range(0, X_dat.size):
        X_dat[i] = X_dat[i] + 10

    Y_dat = dat["Points:1"].values
    for i in range(0, Y_dat.size):
        Y_dat[i] = Y_dat[i] + 4.5
    u_dat = dat["U:0"].values #Velocity x direction
    v_dat = dat["U:1"].values #Velocity y direction
    U_dat = np.sqrt(u_dat**2 + v_dat**2) #U magnitude of velocity
    p_dat = dat["p"].values #pressure

    xy = []
    for i in range(0, X_dat.size):
        x = X_dat[i]
        y = Y_dat[i]
        xy.append([x, y])
        
    xy = np.array(xy)
    XY_data = torch.from_numpy(xy).float().to(device)

    uvp_data = []

    for i in range(0, u_dat.size):
        u = u_dat[i]
        v = v_dat[i]
        p = p_dat[i]
        uvp_data.append([u, v, p])
        
    uvp_data = np.array(uvp_data)
    UVP_data = torch.from_numpy(uvp_data).float().to(device)
    
    Re = 20 # number of reynolds
    nu = 0.5 # kinematic viscosity (m^2/s) 
    U_inf = (Re*nu)/(2*r_cyl*10) # Velocity at inlet of Bounds
    
    max_epochs = args.max_iter # maximum iterations

    random_XY_data, XY_data, random_UVP_data, UVP_data, BC_XY_data, BC_UVP_data = load_data(file_path, N_mes, device)

    # Initialize model
    u_net= PINN_NavierStokes(args.layers).to(args.device)
    # If a pretrained model is provided, load it. Otherwise, initialize the weights
    if args.pretrained_model is not None:
        try:
            u_net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        except Exception as e:
            print(f"Error loading the pretrained model: {e}")
            print("Training the model from scratch...")
            u_net.apply(init_weights)
    else:
        u_net.apply(init_weights)
    
    # If we're not using a pretrained model, then we train the model
    if args.pretrained_model is None:
        # Optimisation
        optimizer = torch.optim.Adam(u_net.parameters(), lr = args.learning_rate)

        # Generate collocation points
        N_x = args.num_X_points
        N_y = args.num_Y_points
        N_bc = args.num_BC_points
        
        XY_pde = colloc_pde_with_mesh(N_x, N_y, x_min, x_max, y_min, y_max, x_center, y_center, r_cyl)
        XY_in = colloc_Xinlet(N_bc, x_min, y_min, y_max)
        XY_out = colloc_Xoutlet(N_bc, x_min, y_min, y_max)
        XY_bot = colloc_Wall_bottom(N_bc, x_min, y_min, y_max)
        XY_top = colloc_Wall_top(N_bc, x_min, y_min, y_max)
        XY_c = colloc_BC_circle(N_bc, x_center, y_center, r_cyl)
        
        # pde loss initialization
        loss = loss_func(u_net, XY_pde, Re, U_inf, XY_in, XY_out, XY_bot, XY_top, XY_c)
        # Total loss initialization
        loss_hard = hard_loss_func(u_net, W_pde, random_XY_data, random_UVP_data, loss)
        
        # Create a CSV file to store the epoch and loss values
        with open('training_log.csv', mode='w', newline='') as csvfile:
            log_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            log_writer.writerow(['Epoch', "Loss_pde", 'Total_loss',  'Learning Rate'])  # Write the header row

            epoch = 0
        
            while (epoch <= max_epochs and loss_hard.item() > 8.5e-7):
                # Perturbation
                if epoch % 5 == 0:
                    XY_pde = colloc_pde_with_mesh(N_x, N_y, x_min, x_max, y_min, y_max, x_center, y_center, r_cyl)
                    XY_in = colloc_Xinlet(N_bc, x_min, y_min, y_max)
                    XY_out = colloc_Xoutlet(N_bc, x_min, y_min, y_max)
                    XY_bot = colloc_Wall_bottom(N_bc, x_min, y_min, y_max)
                    XY_top = colloc_Wall_top(N_bc, x_min, y_min, y_max)
                    XY_c = colloc_BC_circle(N_bc, x_center, y_center, r_cyl)
                # forward and loss
                # loss for pde calculation
                loss = loss_func(u_net, XY_pde, Re, U_inf, XY_in, XY_out, XY_bot, XY_top, XY_c)
                #Total loss (pde + data)
                loss_hard = hard_loss_func(u_net, W_pde, random_XY_data, random_UVP_data, loss)
                # backward
                loss_hard.backward()
                # update
                optimizer.step()
                
                # Update the learning rate scheduler
                #scheduler.step(loss_hard)
                # Update the learning rate scheduler
                #learning_rate = exp_decay_schedule(epoch, initial_lr, decay_rate, plateau_epoch)
                #for param_group in optimizer.param_groups:
                    #param_group['lr'] = learning_rate

                # Write the epoch and loss values to the CSV file
                log_writer.writerow([epoch, loss.item(), loss_hard.item(), args.learning_rate])

                if epoch % 500 == 0:
                    #current_learning_rate = optimizer.param_groups[0]['lr']
                    print(f'epoch: {epoch}, loss_pde: {loss.item()}, total_loss:{loss_hard.item()}, lr = {args.learning_rate}', flush=True)
                
                    # Save the U_NET model every 10,000 steps
                if epoch % 10000 == 0:
                    torch.save(u_net.state_dict(), f'u_net_epoch_{epoch}.pth')
                
                optimizer.zero_grad()
                epoch += 1
    
        # Save your model
        PATH = "u_net_final.pth"
        torch.save(u_net.state_dict(), PATH)

    X_pred = np.hstack((X_dat.flatten()[:,None], Y_dat.flatten()[:,None]))
    X_pred = torch.from_numpy(X_pred).float().to(device)

    # Deep Learning Model Prediction
    u_pred = torch.sqrt(u_net(X_pred)[:,0]**2 + u_net(X_pred)[: ,1]**2)
    p_pred = u_net(X_pred)[:,-1]

    U_pred = u_pred.detach().cpu().numpy().flatten() # Magnitude of Velocity predicted
    P_pred = p_pred.detach().cpu().numpy().flatten() # Pressure predicted
    
    # Plotting results
    print("Results plotting ...")
    plot_magnitude(X_dat, Y_dat, U_dat, 'U magnitude (ground truth)', "pictures/U_magnitude.png")
    plot_magnitude(X_dat.flatten(), Y_dat.flatten(), U_pred, 'U magnitude (predicted with PINN)', "pictures/U_magnitude_predicted.png")
    plot_error(X_dat.flatten(), Y_dat.flatten(), U_dat, U_pred, 'U magnitude Absolute Error ', "pictures/Umagn_ERROR.png")
    plot_magnitude(X_dat, Y_dat, p_dat, 'p (ground truth)', "pictures/pressure.png")
    plot_magnitude(X_dat.flatten(), Y_dat.flatten(), P_pred, 'p (predicted with PINN)', "pictures/pressure_predicted.png")
    plot_error(X_dat.flatten(), Y_dat.flatten(), p_dat, P_pred, 'p Absolute Error ', "pictures/p_ERROR.png")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model configuration
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 96, 96, 96, 96, 96, 96, 3])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data configuration
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, help='Path to the pretrained model', default=None)
    parser.add_argument('--num_X_points', type=int, default=50)
    parser.add_argument('--num_Y_points', type=int, default=50)
    parser.add_argument('--num_BC_points', type=int, default=80)
    parser.add_argument('--max_iter', type=int, default=50000)
    args = parser.parse_args()

    main(args)