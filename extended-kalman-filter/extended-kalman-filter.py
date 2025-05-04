import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import math


class EKF:

    def __init__(self):
        self.dim_state = 4 
        self.u = np.array([[0., 0.]])
        self.dt = 0.1
        self.q = 0.1
        self.F =  self.F()
        self.Q =  self.Q()
        self.H =  self.H()
        self.HR =  self.HR()    
    
    def F(self):

        return np.array([[1., 0., self.dt, 0.],
                        [0., 1., 0., self.dt],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]]
        )

    def Q(self):

        return np.array([[self.dt**3 * self.q / 3., 0., self.dt**2 * self.q / 2., 0.  ],
                        [0, self.dt**3 * self.q / 3., 0, self.dt**2 / 2. ],
                        [self.dt**2 * self.q / 2., 0., self.dt * self.q, 0.],
                        [0., self.dt**2 * self.q / 2., 0., self.dt * self.q]]
        )

    def H(self):

        return np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.]]
        )

    def HR(self):

        return np.array([[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]]                    
        )
        
           
    def predict(self, x: np.ndarray, P: np.ndarray):

        x = self.F @ x
        P = np.matmul(self.F @ P, self.F.T) + self.Q

        return x, P

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray):

        gamma = z - self.H @ x
        S = np.matmul(self.H @ P, self.H.T) + R
        K = np.matmul(P @ self.H.T, np.linalg.inv(S))
        x = x + K @ gamma
        I = np.identity(n = self.dim_state)
        P = (I - np.matmul(K, self.H)) @ P
        
        return x, P

    def radarUpdate(self, x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray):

        z_pred = self.radarCartesianToPolar(x)
        gamma = z - z_pred

        while gamma[1] > math.pi:
            gamma[1] = gamma[1] - 2 * math.pi
        
        while gamma[1] < -math.pi:
            gamma[1] = gamma[1] + 2 * math.pi

        S = np.matmul(self.HR @ P, self.HR.T) + R
        K = np.matmul(P @ self.HR.T, np.linalg.inv(S))
        x = x + K @ gamma

        I = np.identity(n = self.dim_state)
        P = (I - np.matmul(K, self.HR)) @ P

        return x, P

    def radarCartesianToPolar(self, x):

        px = float(x[0])
        py = float(x[1])
        vx = float(x[2])
        vy = float(x[3])

        rho = math.sqrt(px * px + py * py)
        phi = math.atan2(py, px)

        if rho < 0.000001:
            rho = 0.000001
        
        rho_dot = (px * vx + py * vy) / rho
        z_pred = np.array([[rho], [phi], [rho_dot]])

        return z_pred

    def calculateJacobian(self, x):

        px = float(x[0])
        py = float(x[1])
        vx = float(x[2])
        vy = float(x[3])

        c1 = px * px + py * py
        c2 = math.sqrt(c1)
        c3 = c1 * c2

        self.HR[0][0] = px / c2
        self.HR[0][1] = py / c2
        self.HR[0][2] = 0
        self.HR[0][3] = 0
        self.HR[1][0] = -py / c1
        self.HR[1][1] = px / c1
        self.HR[1][2] = 0
        self.HR[1][3] = 0
        self.HR[2][0] = py * (vx * py - vy * px) / c3
        self.HR[2][1] = px * (vy * px - vx * py) / c3
        self.HR[2][2] = px / c2
        self.HR[2][3] = py / c2

        return self.HR
       

def run_filter():
    np.random.seed(0)
    KF = EKF()
    fig, ax = plt.subplots(figsize = (24, 20))

    x = np.array([[1],
                [1],
                [0],
                [0]])
                
    P = np.array([[0.1**2, 0, 0, 0],
                [0, 0.1**2, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 4]])

    #read dataset

    data = open('data2.txt', 'r')
    rows = []

    for line in data:
        row = line.split()
        rows.append(row)

    for row_d in rows:
        sensor = row_d[0]

        if sensor == "L":
            x1 = float(row_d[1])
            y1 = float(row_d[2])
            ti = float(row_d[3])
            x_gt = row_d[4]
            y_gt = row_d[5]
            vx_gt = row_d[6]
            vy_gt = row_d[7]

        if sensor == "R":
            ro = float(row_d[1])
            phi = float(row_d[2])
            ro_dot = float(row_d[3])
            ti = float(row_d[4])
            x_gt = row_d[5]
            y_gt = row_d[6]
            vx_gt = row_d[7]
            vy_gt = row_d[8]
        
        KF.dt = (ti - KF.dt) / 1000000.0
        KF.dt = ti

        x, P = KF.predict(x, P)

        if sensor == "L":
            
            gt = np.array([[x_gt],
                            [y_gt]])
            sigma_z = 0.2

            z = np.array([[float(gt[0]) + np.random.normal(0, sigma_z)],
                        [float(gt[1]) + np.random.normal(0, sigma_z)]])

            z1 = np.array([[float(gt[0]) + np.random.normal(0, sigma_z)],
                        [float(gt[1]) + np.random.normal(0, sigma_z)]])

            R = np.array([[sigma_z**2, 0],
                        [0, sigma_z**2]])
                
            x, P = KF.update(x, P, z, R)

        if sensor == "R":
            
            gt = np.array([[ro * math.cos(phi)],
                            [ro * math.sin(phi)],
                            [0]
                            ])

            z = np.array([[ro],
                            [phi],
                            [ro_dot]
                            ])

            sigma_z = 0.2          

            z1 = np.array([[float(gt[0]) + np.random.normal(0, sigma_z)],
                        [float(gt[1]) + np.random.normal(0, sigma_z)],
                        [float(gt[2]) + np.random.normal(0, sigma_z)]])

            R = np.array([[0.9, 0, 0],
                        [0, 0.0009, 0],
                        [0, 0, 0.9]])
                
            KF.HR = KF.calculateJacobian(x)
            x, P = KF.radarUpdate(x, P, z, R)

        ax.scatter(np.float64(x[0]).tolist(), np.float64(x[1]).tolist(),
                color = 'blue', s = 40, marker = 'x', label = 'track')

        ax.scatter(np.float64(z1[0]).tolist(), np.float64(z1[1]).tolist(),
                color = 'red', s = 40, marker = '.', label = 'measurement')
        
        ax.scatter(np.float64(gt[0]).tolist(), np.float64(gt[1]).tolist(),
                color = 'yellow', s = 40, marker = '+', label = 'GT')            
           
    
        ax.set_xlabel('x [m]', fontsize = 16)
        ax.set_ylabel('y [m]', fontsize = 16)

        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        
        #ax.set_xlim(0, 20)
        #ax.set_ylim(-20, 10)   

        if matplotlib.rcParams['backend'] == 'wxagg':
            mng = plt.get_current_fig_manager()
            mng.frame.Maximize(True)

        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)

        ax.legend(handle_list, label_list, loc = 'lower right',
                    shadow = True, fontsize = 'x-large')
        
        ax.set_title('Sensor Fusion Radar and Lidar data using extended kalman filter', fontsize = 20)
        #animation
        #plt.pause(0.01)
    plt.show()

run_filter()