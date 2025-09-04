Advanced Kalman Filtering and Sensor Fusion

Equation Extended Kalman Filter source from Udemy course https://www.udemy.com/course/advanced-kalman-filtering-and-sensor-fusion

![Screenshot from 2025-05-04 17-57-56](https://github.com/user-attachments/assets/9c513136-6759-48dc-b0df-d570cc405b80)

![Screenshot from 2025-05-04 17-59-14](https://github.com/user-attachments/assets/a106c857-2c4b-4303-8642-a4c9cb9369fc)


Equation Unscented Kalman Filter source from Udemy course https://www.udemy.com/course/advanced-kalman-filtering-and-sensor-fusion

![Screenshot from 2025-05-04 18-00-19](https://github.com/user-attachments/assets/44fd83b6-1114-482e-a136-0027ada17b0f)

![Screenshot from 2025-05-04 18-04-16](https://github.com/user-attachments/assets/bb264002-74dc-40ba-83c7-73536ad3cd0e)

![Screenshot from 2025-05-04 18-05-35](https://github.com/user-attachments/assets/36893a2a-60c9-4965-a8ec-f8bcba6a7afd)

![Screenshot from 2025-05-04 18-07-14](https://github.com/user-attachments/assets/c75d014e-8d45-4dbc-b32d-c8c94c11b0f1)

The Extended Kalman Filter (EKF) linearizes nonlinear functions using Taylor series approximations and Jacobians, while the Unscented Kalman Filter (UKF) uses the Unscented Transform (UT) with sigma points to sample the probability distribution and propagate it through the nonlinear functions, avoiding the need for derivatives. The UKF is generally more accurate for highly nonlinear systems and more robust to initial estimates but can be computationally more expensive than the EKF, which can be faster for mildly nonlinear systems

Results:

Run the same dataset1 using extended and unscented kalman filter

Dataset: Radar_Lidar_1.txt

![extendedkalmanfilter](https://github.com/user-attachments/assets/9c0cbbec-59c8-4e3f-ae50-b97ab1830e9c)

![unscented3](https://github.com/user-attachments/assets/e5b8a54e-e145-407b-807f-7b53e1e83e37)

Results:

Run the same dataset2 using extended and unscented kalman filter

Dataset: Radar_Lidar_2.txt

![extendedkalmanfilter2](https://github.com/user-attachments/assets/6151eeec-5a35-4d76-813a-eb22c143fe5f)


![unscented4](https://github.com/user-attachments/assets/d3f81134-895a-4653-a1c5-1be2aa7d9ca7)
