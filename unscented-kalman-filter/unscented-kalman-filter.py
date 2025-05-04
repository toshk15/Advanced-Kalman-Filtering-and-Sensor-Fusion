import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import math


class UKF:

    def __init__(self):
        self.dt = 0
        self.start_init = 0 
        self.n_x_ = 5
        self.n_aug_ = 7
        self.lambda_ = 3 - self.n_aug_
        self.x_ = np.zeros(self.n_x_)
        self.z_ = np.zeros(self.n_x_)
        self.gt = np.zeros(self.n_x_)
        self.P_ = np.zeros([self.n_x_, self.n_x_])
        self.Xsig_pred_ = np.zeros([self.n_x_, 2 * self.n_aug_ + 1])
        self.weights_ = self.W()
        self.std_a_ = 1.5
        self.std_yawdd_ = 0.5
        self.time_us_ = 0.0
        self.std_laspx_ = 0.15
        self.std_laspy_ = 0.15
        self.std_radr_ = 0.3
        self.std_radphi_ = 0.03
        self.std_radrd_ = 0.3
        
    def W(self):

        j = 2 * self.n_aug_
        self.weights_ = np.zeros(2 * self.n_aug_ + 1)
        self.weights_[0] = self.lambda_ / (self.lambda_ + self.n_aug_)
        for i in range(j):
            self.weights_[i + 1] = 1 / (2 * self.lambda_ + 2 * self.n_aug_)
        return self.weights_
    

    def predict(self):

        x_aug = np.zeros(self.n_aug_)
        x_aug[0] = self.x_[0]
        x_aug[1] = self.x_[1]
        x_aug[2] = self.x_[2]
        x_aug[3] = self.x_[3]
        x_aug[4] = self.x_[4]
        x_aug[5] = 0
        x_aug[6] = 0

        P_aug = np.zeros([self.n_aug_, self.n_aug_])
        P_aug[:5, :5] = self.P_
        P_aug[5, 5] = self.std_a_ * self.std_a_
        P_aug[6, 6] = self.std_yawdd_ * self.std_yawdd_

        Xsig_aug = np.zeros([self.n_aug_, 2 * self.n_aug_ + 1])

        L = np.linalg.cholesky(P_aug)
        L *= math.sqrt(3)

        Xsig_aug[0: self.n_aug_, 0] = x_aug

        for i in range(self.n_aug_):
            Xsig_aug[:, i + 1] = x_aug + L[:, i]
            Xsig_aug[:, i + 1 + self.n_aug_] = x_aug - L[:, i]
        
        #predict sigma points

        for i in range(2 * self.n_aug_ + 1):
            px = Xsig_aug[0, i]
            py = Xsig_aug[1, i]
            v = Xsig_aug[2, i]
            yaw = Xsig_aug[3, i]
            yawd = Xsig_aug[4, i]
            nu_a = Xsig_aug[5, i]
            nu_yawdd = Xsig_aug[6, i]
            v_pred = v
            yaw_pred = yaw
            yawd_pred = yawd

            if abs(yawd) > 0.0001:
                px += v / yawd * (math.sin(yaw + yawd * self.dt) - math.sin(yaw))
                py += v / yawd * (-1 * math.cos(yaw + yawd * self.dt) + math.cos(yaw))
                yaw_pred += yawd * self.dt
            else:
                px += v * math.cos(yaw) * self.dt
                py += v * math.sin(yaw) * self.dt

            px += 0.5 * self.dt * self.dt * math.cos(yaw) * nu_a
            py += 0.5 * self.dt * self.dt * math.sin(yaw) * nu_a
            v_pred += self.dt * nu_a
            yaw_pred += 0.5 * self.dt * self.dt * nu_yawdd
            yawd_pred += self.dt * nu_yawdd

            self.Xsig_pred_[0, i] = px
            self.Xsig_pred_[1, i] = py
            self.Xsig_pred_[2, i] = v_pred
            self.Xsig_pred_[3, i] = yaw_pred
            self.Xsig_pred_[4, i] = yawd_pred
        
        #predict covariance
        self.x_ = np.zeros(self.n_x_)

        for i in range(2 * self.n_aug_ + 1):
            self.x_ = self.x_ + (self.weights_[i] * self.Xsig_pred_[:, i])
        
        self.P_ = np.zeros([self.n_x_ , self.n_x_])

        for i in range(2 * self.n_aug_ + 1):
            x_diff = self.Xsig_pred_[:, i] - self.x_
            x_diff = np.reshape(x_diff, (5, 1))

            while x_diff[3] > math.pi:
                x_diff[3] -= 2.0 * math.pi
            
            while x_diff[3] < -math.pi:
                x_diff[3] += 2.0 * math.pi
            
            self.P_ += self.weights_[i] * x_diff * x_diff.T

    def updateLidar(self):
            
        n_z = 2
        Zsig = np.zeros([n_z, 2 * self.n_aug_ + 1])
        z_pred = np.zeros(n_z)
        S = np.zeros([n_z, n_z])
        Tc = np.zeros([self.n_x_ , n_z])

        for i in range(2 * self.n_aug_ + 1):
            px = self.Xsig_pred_[0, i]
            py = self.Xsig_pred_[1, i]
            Zsig[0, i] = px
            Zsig[1, i] = py

        for i in range(2 * self.n_aug_ + 1):
            z_pred = z_pred + self.weights_[i] * Zsig[:, i]

        #innovation covariance matrix
        for i in range(2 * self.n_aug_ + 1):
            z_diff = Zsig[:, i] - z_pred

            while z_diff[1] > math.pi:
                z_diff[1] -= 2.0 * math.pi

            while z_diff[1] < -math.pi:
                z_diff[1] += 2.0 * math.pi

            z_diff = np.reshape(z_diff, (2, 1))
            S = S + self.weights_[i] * z_diff * z_diff.T

        R = np.array([[self.std_laspx_ * self.std_laspx_, 0], 
                        [0, self.std_laspy_ * self.std_laspy_]])
        S = S + R

        for i in range(2 * self.n_aug_ + 1):
            z_diff = Zsig[:, i] - z_pred

            while z_diff[1] > math.pi:
                z_diff[1] -= 2.0 * math.pi
            
            while z_diff[1] < -math.pi:
                z_diff[1] += 2.0 * math.pi
            
            z_diff = np.reshape(z_diff, (2, 1))
            x_diff = self.Xsig_pred_[:, i] - self.x_

            while x_diff[3] > math.pi:
                x_diff[3] -= 2.0 * math.pi
            
            while x_diff[3] < -math.pi:
                x_diff[3] += 2.0 * math.pi

            x_diff = np.reshape(x_diff, (5, 1))
            Tc += self.weights_[i] * x_diff * z_diff.T

        K = Tc @ np.linalg.inv(S)
        z = np.zeros([2, 1])
        z[0] = self.z_[0]
        z[1] = self.z_[1]
        z_pred = np.reshape(z_pred, (2, 1))
        z_diff = z - z_pred

        while z_diff[1] > math.pi:
            z_diff[1] -= 2.0 * math.pi
            
        while z_diff[1] < -math.pi:
            z_diff[1] += 2.0 * math.pi
        
        z_diff = np.reshape(z_diff, (2, 1))
        self.x_ = np.reshape(self.x_, (5, 1))

        self.x_ += K @ z_diff
        self.P_ -= K @ S @ K.T

    def updateRadar(self):
            
        n_z = 3
        Z_sig = np.zeros([n_z, 2 * self.n_aug_ + 1])
        z_pred = np.zeros(n_z)
        S = np.zeros([n_z, n_z])
        Tc = np.zeros([self.n_x_, n_z])

        for i in range(2 * self.n_aug_ + 1):
            px = self.Xsig_pred_[0, i]
            py = self.Xsig_pred_[1, i]
            v = self.Xsig_pred_[2, i]
            yaw = self.Xsig_pred_[3, i]

            v1 = math.cos(yaw) * v
            v2 = math.sin(yaw) * v

            Z_sig[0, i] = math.sqrt(px * px + py * py)
            Z_sig[1, i] = math.atan2(py, px)
            Z_sig[2, i] = (px * v1 + py * v2) / math.sqrt(px * px + py * py)
        
        for i in range(2 * self.n_aug_ + 1):
            z_pred = z_pred + self.weights_[i] * Z_sig[:, i]
        
        for i in range(2 * self.n_aug_ + 1):
            z_diff = Z_sig[:, i] - z_pred

            while z_diff[1] > math.pi:
                z_diff[1] -= 2.0 * math.pi
            
            while z_diff[1] < -math.pi:
                z_diff[1] += 2.0 * math.pi

            z_diff = np.reshape(z_diff, (3, 1))
            S = S + self.weights_[i] * z_diff * z_diff.T

        R = np.array([[self.std_radr_ * self.std_radr_, 0, 0],
                        [0, self.std_radphi_ * self.std_radphi_, 0],
                        [0, 0, self.std_radr_ * self.std_radr_]])
        
        S = S + R

        #update state mean and covariance

        for i in range(2 * self.n_aug_ + 1):
            x_diff = self.Xsig_pred_[:, i] - self.x_

            while x_diff[3] > math.pi:
                x_diff[3] -= 2.0 * math.pi
            
            while x_diff[3] < -math.pi:
                x_diff[3] += 2.0 * math.pi

            x_diff = np.reshape(x_diff, (5, 1))
            z_diff = Z_sig[:, i] - z_pred

            while z_diff[1] > math.pi:
                z_diff[1] -= 2.0 * math.pi
            
            while z_diff[1] < -math.pi:
                z_diff[1] += 2.0 * math.pi

            z_diff = np.reshape(z_diff, (3, 1))
            Tc += self.weights_[i] * x_diff * z_diff.T

        Si = np.linalg.inv(S)
        Tc = np.reshape(Tc, (5, 3))
        K = Tc @ Si

        z = np.zeros(3)
        z[0] = self.z_[0]
        z[1] = self.z_[1]
        z[2] = self.z_[2]

        z_diff = z - z_pred

        while z_diff[1] > math.pi:
            z_diff[1] -= 2.0 * math.pi
            
        while z_diff[1] < -math.pi:
            z_diff[1] += 2.0 * math.pi

        z_diff = np.reshape(z_diff, (3, 1))
        self.x_ = np.reshape(self.x_, (5, 1))

        self.x_ += K @ z_diff
        self.P_ -= K @ S @ K.T

def run_filter():
    np.random.seed(0)
    ukf = UKF()
    fig, ax = plt.subplots(figsize = (24, 20))

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

        if ukf.start_init == 0:
            ukf.start_init = 1
            ts = ti

            if sensor == "L":
                ukf.x_ = np.array([[x1],
                                    [y1],
                                    [0],
                                    [0],
                                    [0]])
                
                ukf.P_ = np.array([[0.15, 0, 0, 0, 0],
                                    [0, 0.15, 0, 0, 0],
                                    [0, 0, 0.3, 0, 0],
                                    [0, 0, 0, 0.3, 0],
                                    [0, 0, 0, 0, 0.3]])
                
            if sensor == "R":
                ukf.x_ = np.array([[ro * math.cos(phi)],
                                    [ro * math.sin(phi)],
                                    [0],
                                    [0],
                                    [0]])
                
                ukf.P_ = np.array([[0.3, 0, 0, 0, 0],
                                    [0, 0.3, 0, 0, 0],
                                    [0, 0, 0.3, 0, 0],
                                    [0, 0, 0, 0.3, 0],
                                    [0, 0, 0, 0, 0.3]])
        else:
            ukf.dt = (ti - ts) / 1000000.0
            ts = ti

            ukf.predict()
            if sensor == "L":
                ukf.z_ = np.array([[x1],
                                [y1],
                                [0],
                                [0],
                                [0]])
                ukf.gt = np.array([[x1],
                                [y1],
                                [0],
                                [0],
                                [0]])
                
                ukf.updateLidar()
        
            if sensor == "R":
                ukf.z_ = np.array([[ro],
                                        [phi],
                                        [ro_dot],
                                        [0],
                                        [0]])
                ukf.gt = np.array([[ro * math.cos(phi)],
                                        [ro * math.sin(phi)],
                                        [0],
                                        [0],
                                        [0]])
                
                ukf.updateRadar()
        ax.scatter(np.float64(ukf.x_[0]).tolist(), np.float64(ukf.x_[1]).tolist(),
                color = 'orange', s = 40, marker = 'x', label = 'track')
    
        ax.scatter(np.float64(ukf.gt[0]).tolist(), np.float64(ukf.gt[1]).tolist(),
                color = 'black', s = 40, marker = '+', label = 'Ground truth')
    
        ax.set_xlabel('x [m]', fontsize = 16)
        ax.set_ylabel('y [m]', fontsize = 16)
        
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)

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
        
        ax.set_title('Sensor Fusion using uncented kalman filter', fontsize = 20)
        #animation
        #plt.pause(0.01)
    plt.show()

run_filter()