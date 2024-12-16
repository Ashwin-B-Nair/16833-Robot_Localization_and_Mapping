'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0,1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad % (2 * np.pi)  
    if angle_rad > np.pi:
        angle_rad -= 2 * np.pi  
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2
    landmarks = np.zeros((2*k, 1))
    landmarks_cov = np.zeros((2*k, 2*k))
    
    for i in range(k):
        bearing = init_measure[2*i,0]
        range_val = init_measure[2*i + 1,0]
        
        #Initial landmark positions
        x = init_pose[0,0] + range_val*np.cos(init_pose[2,0] + bearing)
        y = init_pose[1  ,0] + range_val*np.sin(init_pose[2,0] + bearing)
        
        #Add to landmark state vector
        landmarks[2*i:2*i+2] = np.array([[x],[y]])

        #Transforming polar coordinates to cartesian space
        J_p_to_c = np.array([[np.cos(init_pose[2,0] + bearing), -range_val * np.sin(init_pose[2,0] + bearing)], 
                             [np.sin(init_pose[2,0] + bearing), range_val * np.cos(init_pose[2,0] + bearing)]])
        
        #Propogating uncertainity 
        landmark_cov = J_p_to_c @ init_measure_cov @ J_p_to_c.T
        
        #Finding J_pose to represent how changes in the robot's pose affects the landmark's position estimate.
        J_pose = np.array([[1, 0 , -range_val * np.sin(init_pose[2,0] + bearing)],
                          [0, 1, range_val * np.cos(init_pose[2,0] + bearing)]])
        
        #Propagating uncertainity form robot pose to landmark position estimate 
        #Accounting for how uncertainty in the robot's pose contributes to uncertainty in the landmark's position.  
        landmark_cov += J_pose @ init_pose_cov @ J_pose.T
        
        landmarks_cov[2*i:2*i+2, 2*i:2*i+2] = landmark_cov
      
    return k, landmarks, landmarks_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    #Extracting the robot pose
    x, y, theta = X[0,0], X[1,0], X[2,0]

    d, alpha = control[0,0], control[1,0]
    
    #Predicting new pose
    x_new = x + d*np.cos(theta)
    y_new = y + d*np.sin(theta)
    theta_new = theta + alpha

    #Updating X with new pose
    X_pre = X.copy()
    X_pre[0:3] = np.array([[x_new], [y_new], [theta_new]])    
    
    # Compute Jacobian G (represents how changes in state affect the prediction)
    '''Only G[0,2] and G[1,2] are explicitly set to non-zero, non-one values because 
    they represent the non-linear relationship between the robot's orientation 
    and its position change.
    The landmark part of G (the lower-right 2k x 2k submatrix) remains an identity matrix 
    (diagonal 1s, rest 0s) because landmarks are assumed to be stationary. 
    Their positions in the next state are expected to be the same as in the current state.'''
    
    G = np.eye(3 + 2*k)                    #Initialize as identify matrix
    G[0, 2] = -d * np.sin(theta + alpha)   #dx/d(theta)
    G[1, 2] = d * np.cos(theta + alpha)    #dy/d(theta)
    
    # Construct process noise covariance Q
    Q = np.zeros((3 + 2*k, 3 + 2*k))
    Q[0:3, 0:3] = control_cov
    
    # Update covariance
    P_pre = G @ P @ G.T + Q
    
    return X_pre, P_pre


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    x, y, theta = X_pre[0,0], X_pre[1,0], X_pre[2,0]

    H = np.zeros((2*k, 3 + 2*k))   #Combined Hp and Hl matrix (from theory section)
    z_exp = np.zeros((2*k, 1))     #All landmarks (x and y for each)
    
     
    for i in range(k):  #Looping through each landmark
        # Extracting landmark position
        lx, ly = X_pre[3 + 2*i, 0], X_pre[3 + 2*i + 1, 0]
        
        # Calculate expected range and bearing
        dx, dy = lx - x, ly - y
        q = dx**2 + dy**2
        r = np.sqrt(q)
        
        bearing = warp2pi(np.arctan2(dy, dx) - theta)
        
        z_exp[2*i:2*i+2] = np.array([[bearing], [r]])
         
        #Compute Jacobian H
        #computing Hp below
        H[2*i:2*i+2, 0:3] = np.array([[dy/q, -dx/q, -1],[-dx/r, -dy/r,0]])
        
        #computing Hl below
        H[2*i:2*i+2, 3 +2*i:3+2*i+2 ] = np.array([[-dy/q, dx/q],[dx/r, dy/r]])
        
    z = measure
    diff = z - z_exp
    
    for i in range(k):
        diff[2*i,0] = warp2pi(diff[2*i,0])
    
    # Constructing measurement noise covariance R using kronecker product
    R = np.kron(np.eye(k), measure_cov)
        
    #Calculating Kalman Gain
    K = P_pre @ H.T @ np.linalg.inv(H@P_pre@H.T + R)
    
    X = X_pre + K @ diff 
                      
    I = np.eye(3 + 2*k)
    P = (I - K @ H) @ P_pre
    return X, P


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)
    
    for i in range(k):
        l_est = X[3+2*i:3+2*i+2, 0]  # Estimated landmark position
        l_true_i = l_true[2*i:2*i+2]  # True landmark position
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(l_est - l_true_i)
        
        # Mahalanobis distance
        landmark_cov = P[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
        mahalanobis_dist = np.sqrt((l_est - l_true_i).T @ np.linalg.inv(landmark_cov) @ (l_est - l_true_i))
        
        print(f"Landmark {i+1}:")
        print(f"  Euclidean distance: {euclidean_dist:.4f}")
        print(f"  Mahalanobis distance: {mahalanobis_dist:.4f}")
    
    def is_inside_ellipse(true_pos, est_pos, cov):
        diff = true_pos - est_pos
        return (diff.T @ np.linalg.inv(cov) @ diff) <= 9.21  # 9.21 is chi-square value for 2 DOF at 99% confidence

    for i in range(k):
        l_est = X[3+2*i:3+2*i+2, 0]
        l_true_i = l_true[2*i:2*i+2]
        landmark_cov = P[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
        
        if is_inside_ellipse(l_true_i, l_est, landmark_cov):
            print(f"Landmark {i+1} is inside its 3-sigma ellipse")
        else:
            print(f"Landmark {i+1} is outside its 3-sigma ellipse")


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("C:/Users/ashwi/Documents/CMU/FALL_2024/SLAM/SLAM_Assigment_2/problem_set/problem_set/data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # print(f"initial P:",P)
    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)
    # print(f"FINAL!!!!!!!!!:",P)
    
    print(f"Final landmark positions: {X[3:]}")
    print(f"Final robot pose: {X[0:3]}")
    print(f"Average landmark uncertainty: {np.mean([np.trace(P[3+2*i:5+2*i, 3+2*i:5+2*i]) for i in range(k)])}")

if __name__ == "__main__":
    main()
