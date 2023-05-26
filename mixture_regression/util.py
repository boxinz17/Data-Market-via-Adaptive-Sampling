import numpy as np
import time

def model_eval(w, val_data):
    # compute the validation loss
    # params:
    #   w: d-dimensional numpy array the parameter
    #   val_data: (X_val, y_val), where X_val is n_val x d, y_val is d-dimensional
    X_val, y_val = val_data
    n_val = len(y_val)

    return (np.linalg.norm(y_val - np.matmul(X_val, w)) ** 2) / (2*n_val)

def model_updates(w, train_data):
    # return the model updates
    # params:
    #   w: d-dimensional numpy array the parameter
    #   train_data: (X_train, y_train), where X_train is n_train x d, y_train is d-dimensional
    X_train, y_train = train_data
    n_train = len(y_train)

    return (1 / n_train) * np.matmul(X_train.T, y_train - np.matmul(X_train, w))

def train_FedAvg_uniform_mr(providers_train_list, val_data, w_true, step_size_global, batch_size, n_commun):
    # number of providers
    n_providers = len(providers_train_list)
    
    # intialize the number of access for each data provider
    n_access = np.zeros(n_providers)

    util_list = []  # list storing validation lost
    est_err_list = []  # list storing estimation error

    # dimension of parameters
    d = providers_train_list[0][0].shape[1]

    # initialize the parameters
    w = -np.ones(d)

    start_time = time.time()
    for t in range(n_commun):
        # compute the estimation error
        est_err_list.append(np.linalg.norm(w-w_true))

        # compute the utility value
        util = -model_eval(w, val_data)
        util_list.append(util)

        # print out useful information
        if t == 0 or (t+1) % 100 == 0:
            print('communication round: {} | validation loss: {} | estimation error: {:.3f}'.format(t+1, -util, est_err_list[-1]))
            print(n_access)

        # sample providers
        chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=False)
        #chosen_providers = [4, 14, 19, 21, 27, 28, 29, 30, 35, 38]

        updates = []  # store the local updates
        for i in chosen_providers:
            n_access[i] += 1
            local_update = model_updates(w, providers_train_list[i])
            updates.append(local_update)

        # make updates of the global model by FedAvg
        g = updates[0]
        for i in range(1, batch_size):
            g += updates[i]
        g /= batch_size
        w += step_size_global * g

    end_time = time.time()
    time_used = end_time - start_time

    return w, n_access, util_list, est_err_list, time_used

def OMD_solver(p, u, lr, alpha):
    n = len(p)
    log_p_new = np.log(p) + lr * u
    log_p_new_sorted = np.sort(log_p_new)
    p_new_sorted = np.exp(log_p_new_sorted)

    i_star = 0
    for i in range(n, 0, -1):
        if log_p_new_sorted[i-1] + np.log(1 - alpha * (i-1) / n) <= np.log(alpha/n) + np.log(p_new_sorted[i-1:].sum()):
            i_star = i+1
            break
    
    ss = p_new_sorted[i_star-1:].sum()
    ss_log = np.log(ss)
    p_new_argsort = np.argsort(log_p_new)
    p_hat = np.zeros(n)
    for i in range(n):
        if i+1 < i_star:
            p_hat[p_new_argsort[i]] = alpha / n
        else:
            p_hat[p_new_argsort[i]] = np.exp( log_p_new[p_new_argsort[i]] + np.log(1 - alpha * (i_star-1) / n) - ss_log )
    
    return p_hat / p_hat.sum()

def train_FedAvg_OSMD_mr(providers_train_list, val_data, w_true, step_size_global, batch_size, n_commun, learning_rate, alpha):
    # number of providers
    n_providers = len(providers_train_list)
    
    # intialize the number of access for each data provider
    n_access = np.zeros(n_providers)

    util_list = []  # list storing validation lost
    est_err_list = []  # list storing estimation error

    # dimension of parameters
    d = providers_train_list[0][0].shape[1]

    # initialize the parameters
    w = -np.ones(d)

    # initialize the sampling distribution as uniform
    prob_sampling = np.ones(n_providers)
    prob_sampling /= prob_sampling.sum()  # initialize the sampling distribution as uniform
    u_hat = np.zeros(n_providers)

    start_time = time.time()
    for t in range(n_commun):
        # compute the estimation error
        est_err_list.append(np.linalg.norm(w-w_true))

        # compute the utility value
        util = -model_eval(w, val_data)
        util_list.append(util)

        # print out useful information
        if t == 0 or (t+1) % 100 == 0:
            print('communication round: {} | validation loss: {} | estimation error: {:.3f}'.format(t+1, -util, est_err_list[-1]))
            print(n_access)
            print(prob_sampling)
            print(u_hat)

        # sample providers
        chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=True, p=prob_sampling)

        updates = []  # store the local updates
        u_hat = np.zeros(n_providers)  # reset the u_hat vector
        for i in chosen_providers:
            n_access[i] += 1
            local_update = model_updates(w, providers_train_list[i])
            updates.append(local_update)

            w_new = w + step_size_global * local_update
            util_new = -model_eval(w_new, val_data)
            u_hat[i] = (util_new - util) / (batch_size * prob_sampling[i])

        # make updates of the global model by FedAvg
        g = updates[0]
        for i in range(1, batch_size):
            g += updates[i]
        g /= batch_size
        w += step_size_global * g

        prob_sampling = OMD_solver(prob_sampling, u_hat, learning_rate, alpha)

    end_time = time.time()
    time_used = end_time - start_time

    return w, n_access, util_list, est_err_list, time_used

def train_DSV_mr(providers_train_list, val_data, step_size_global, back_check, c_tol, max_n_perm):
    # params:
    #   back_check:  integer, the gap to check convergence of DSV
    #   c_tol: float, tolerance to decide if DSV has converged
    #   max_n_perm: integer, maximum number of permutations

    # number of providers
    n_providers = len(providers_train_list)

    # dimension of parameters
    d = providers_train_list[0][0].shape[1]

    DSV_list = []  # List to store the DSV values
    DSV = np.zeros(n_providers)  # initilaize the Data Shapley values
    DSV_convg_measure = []  # check if DSV has converged

    start_time = time.time()
    t  = 0
    while True:
        # check if the number of permutations exceeds the maximum threshold
        t += 1
        if t > max_n_perm:
            break

        # shuffle all the training samples
        shuffled_seq = np.arange(n_providers)
        np.random.shuffle(shuffled_seq)

        # initialize the parameters
        w = -np.ones(d)

        for count_i, i in enumerate(shuffled_seq):
            # make local computation
            local_update = model_updates(w, providers_train_list[i])
            w += step_size_global * local_update

            if count_i == 0:
                # intialize the value function
                v = -model_eval(w, val_data)
                continue

            # compute the local utility value
            v_new = -model_eval(w, val_data)

            # update the DSV
            DSV[i] = ((t-1) / t) * DSV[i] + (1 / t) * (v_new - v)

            # update the value function
            v = v_new

        # append the DSV to DSV list
        DSV_list.append(DSV.copy())

        # print out the DSV
        if t == 1 or t % 100 == 0:
            print(DSV)

        # compute the convergence criterion of DSV
        if t > back_check:
            s = 0.0
            count = 0
            for i in range(len(DSV)):
                if abs(DSV_list[-1][i]) > 0.0:
                    count += 1
                    s += abs(DSV_list[-1][i] - DSV_list[-1-back_check][i]) / abs(DSV_list[-1][i])
            s /= count
            DSV_convg_measure.append(s)

            if s < c_tol:
               print("The permutation round: {} | convergence criterion: {:.3f}".format(t, DSV_convg_measure[-1]))
               break

        # print out useful information
        if t == 1 or t % 100 == 0:
            if len(DSV_convg_measure) > 0:
                print("The permutation round: {} | convergence criterion: {:.3f}".format(t, DSV_convg_measure[-1]))
            else:
                print("The permutation round: {}".format(t))

    end_time = time.time()
    time_used = end_time - start_time

    return DSV_list, DSV_convg_measure, DSV, time_used

def train_FedAvg_FedDSV_mr(providers_train_list, val_data, w_true, step_size_global, batch_size, n_commun, back_check, c_tol):
    # params:
    #   c_tol: float, convgernece tolerance to decide if DSV has converged
    #   back_check: integer, the gap to check convergence of DSV

    # number of providers
    n_providers = len(providers_train_list)

    # dimension of parameters
    d = providers_train_list[0][0].shape[1]

    # initialize the parameters
    w = -np.ones(d)

    n_access = np.zeros(n_providers)

    util_list = []  # list storing validation lost
    est_err_list = []  # list storing test accuracy
    start_time = time.time()
    FedDSV = np.zeros(n_providers)  # initilaize the Federated Data Shapley values

    for t in range(n_commun):
        # compute the estimation error of the global parameter
        est_err_list.append(np.linalg.norm(w-w_true))

        # compute the utility value of the global parameter
        util = -model_eval(w, val_data)
        util_list.append(util)

        # print out useful information
        if t == 0 or (t+1) % 100 == 0:
            print('communication round: {} | validation loss: {} | estimation error: {:.3f}'.format(t+1, -util, est_err_list[-1]))
            print(n_access)
            print(FedDSV)

        # sample providers
        chosen_providers = np.random.choice(np.arange(n_providers), size=batch_size, replace=False)

        updates = []  # store the local updates
        for i in chosen_providers:
            n_access[i] += 1
            local_update = model_updates(w, providers_train_list[i])
            updates.append(local_update)

        # compute the Federated Data Shapley value of this round
        FedDSV_t = np.zeros(n_providers)
        DSV_convg_measure = []  # list of convergence criterion of DSV
        DSV_list = []  # list storing DSV
        n_perm = 0  # number of permutations
        chosen_providers_shuffled = np.arange(batch_size)
        while True:
            n_perm += 1

            # shuffle all chosen providers
            np.random.shuffle(chosen_providers_shuffled)

            # initialize the value function
            U_prev = util

            # update the Fed DSV of this round
            for i in chosen_providers_shuffled:
                w_temp = w + step_size_global * updates[i]
                U_new = -model_eval(w_temp, val_data)
                FedDSV_t[chosen_providers[i]] =  ((n_perm-1) / n_perm) * FedDSV_t[chosen_providers[i]] + (1 / n_perm) * (U_new - U_prev)
                U_prev = U_new

            # append the DSV to DSV list
            DSV_list.append(FedDSV_t.copy())

            if n_perm > back_check:
                s = 0.0
                count = 0
                for i in range(batch_size):
                    if abs(DSV_list[-1][chosen_providers[i]]) > 0.0:
                        count += 1
                        s += abs(DSV_list[-1][chosen_providers[i]] - DSV_list[-1-back_check][chosen_providers[i]]) / abs(DSV_list[-1][chosen_providers[i]])
                s /= count
                DSV_convg_measure.append(s)

                if s < c_tol:
                    print("Fed DSV of round {} converged | # permutations: {} | convergence critertion: {}".format(t+1, n_perm, s))
                    break

        # update the cumulative Fed DSV
        FedDSV += FedDSV_t

        # make updates of the global model by FedAvg
        g = updates[0]
        for i in range(1, batch_size):
            g += updates[i]
        g /= batch_size
        w += step_size_global * g

    end_time = time.time()
    time_used = end_time - start_time

    return w, n_access, util_list, est_err_list, time_used, FedDSV