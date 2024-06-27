import numpy as np
import Shapelet.auto_pisd as auto_pisd
import Shapelet.pst_support_method as pstsm
import Shapelet.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial
import pickle


class ShapeletDiscover():
    def __init__(self, window_size=20, num_pip=0.4, processes=4, len_of_ts=None, dim=1):
        self.window_size = window_size
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = len_of_ts
        self.list_labels = None
        self.dim=dim
        self.processes = processes

    # save list_group_ppi with pickle
    def save_shapelet_candidates(self, path="store/s1.pkl"):
        file = open(path, 'wb')
        pickle.dump(self.list_group_ppi, file)
        file.close()

    # load shapelet information from disk
    def load_shapelet_candidates(self, path="store/s1.pkl"):
        file = open(path, 'rb')
        ppi = pickle.load(file)
        if ppi is not None:
            self.list_group_ppi = ppi
        file.close()

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_shapelet_info(self, number_of_shapelet, p=0.0, pi=0.0):
        if number_of_shapelet == 0:
            number_of_shapelet = 1

        list_shapelet = None
        for i in range(len(self.list_group_ppi)):
            list_ppi = np.concatenate(self.list_group_ppi[i])
            list_group_shapelet = pstsm.find_c_shapelet_non_overlab(list_ppi, number_of_shapelet, p=p, p_inner=pi, len_ts=self.len_of_ts)
            list_group_shapelet = np.asarray(list_group_shapelet)
            list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
            if list_shapelet is None:
                list_shapelet = list_group_shapelet
            else:
                list_shapelet = np.concatenate((list_shapelet, list_group_shapelet), axis=0)

        return list_shapelet

    def get_shapelet_info_v1(self, number_of_shapelet):
        if number_of_shapelet == 0:
            number_of_shapelet = 1

        list_shapelet = None
        for i in range(len(self.list_group_ppi)):
            for d in range(self.dim):
                list_ppi = self.list_group_ppi[i][d]
                list_group_shapelet = pstsm.find_c_shapelet_non_overlab(list_ppi, number_of_shapelet)
                list_group_shapelet = np.asarray(list_group_shapelet)
                list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
                if list_shapelet is None:
                    list_shapelet = list_group_shapelet
                else:
                    list_shapelet = np.concatenate((list_shapelet,list_group_shapelet),axis=0)

        return list_shapelet

    def find_ppi(self, i, l, d):
        print("discovery %s - %s - %s" % (i, l, d))
        list_result = []
        ts_pos = self.group_train_data_pos[l][i]
        pdm = {}
        t1 = self.group_train_data[l][i][d]
        pdm[i * 100000 + i] = np.zeros((0, 0))

        time1 = time.time()
        for p in range(len(self.train_data)):
            t2 = self.train_data[p][d]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
            pdm[ts_pos * 100000 + p] = matrix_1
        print("T1: %s" % (time.time() - time1))
        time1 = time.time()
        for j in range(len(self.group_train_data_piss[l][i][d])):
            ts_pis = self.group_train_data_piss[l][i][d][j]
            ts_ci_pis = self.group_train_data_ci_piss[l][i][d][j]
            # Calculate subdist with all time series
            list_dist = []
            for p in range(len(self.train_data)):
                if p == ts_pos:
                    list_dist.append(0)
                else:
                    matrix = pdm[ts_pos * 100000 + p]
                    ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)
                    ts_2_ci = self.train_data_ci[p][d]
                    pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                    dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos,
                                                   self.list_end_pos, ts_ci_pis, pcs_ci_list)
                    list_dist.append(dist)

            # Calculate best information gain
            ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
            # ig = 0

            # time series position, start_pos, end_pos, inforgain, label, dim
            ppi = [ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l], d]
            list_result.append(ppi)
        print("T2: %s" % (time.time() - time1))
        return list_result

    def find_ppi2(self, d, l, i):
        print("discovery %s - %s - %s" % (i, l, d))
        list_result = []
        ts_pos = self.group_train_data_pos[l][i]
        pdm = {}
        t1 = self.group_train_data[l][i][d]
        pdm[i * 100000 + i] = np.zeros((0, 0))

        time1 = time.time()
        for p in range(len(self.train_data)):
            t2 = self.train_data[p][d]
            matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
            pdm[ts_pos * 100000 + p] = matrix_1
        print("T1: %s" % (time.time() - time1))
        time1 = time.time()
        for j in range(len(self.group_train_data_piss[l][i][d])):
            ts_pis = self.group_train_data_piss[l][i][d][j]
            ts_ci_pis = self.group_train_data_ci_piss[l][i][d][j]
            # Calculate subdist with all time series
            list_dist = []
            for p in range(len(self.train_data)):
                if p == ts_pos:
                    list_dist.append(0)
                else:
                    matrix = pdm[ts_pos * 100000 + p]
                    ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)
                    ts_2_ci = self.train_data_ci[p][d]
                    pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                    dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos,
                                                   self.list_end_pos, ts_ci_pis, pcs_ci_list)
                    list_dist.append(dist)

            # Calculate best information gain
            ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
            # ig = 0

            # time series position, start_pos, end_pos, inforgain, label, dim
            ppi = [ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l], d]
            list_result.append(ppi)
        print("T2: %s" % (time.time() - time1))
        return list_result

    def extract_candidate(self, train_data):
        # Extract shapelet candidate
        time1 = time.time()
        self.train_data_piss = [[]for i in range(len(train_data))]
        p = multiprocessing.Pool(processes=self.processes)
        for i in range(len(train_data)):
            time_series = train_data[i]
            temp_ppi = p.map(partial(auto_pisd.auto_piss_extractor, time_series=time_series, num_pip=self.num_pip, j=i), range(self.dim))
            self.train_data_piss[i] = temp_ppi

        ci_return = [auto_pisd.auto_ci_extractor(train_data[i], self.train_data_piss[i]) for i in range(len(train_data))]
        self.train_data_ci = [ci_return[i][0] for i in range(len(ci_return))]
        self.train_data_ci_piss = [ci_return[i][1] for i in range(len(ci_return))]

        time1 = time.time() - time1
        print("extracting time: %s" % time1)


    def discovery(self, train_data, train_labels, flag=1):
        time2 = time.time()
        self.train_data = train_data
        self.train_labels = train_labels

        self.len_of_ts = len(train_data[0][0])
        self.list_labels = np.unique(train_labels)

        self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
        self.list_end_pos = np.ones(self.len_of_ts, dtype=int) * (self.window_size * 2 + 1)
        for i in range(self.window_size):
            self.list_end_pos[-(i + 1)] -= self.window_size - i
        for i in range(self.window_size - 1):
            self.list_start_pos[i] += self.window_size - i - 1

        # Divide time series into group of label
        self.group_train_data = [[] for i in self.list_labels]
        self.group_train_data_pos = [[] for i in self.list_labels]
        self.group_train_data_piss = [[] for i in self.list_labels]
        self.group_train_data_ci_piss = [[] for i in self.list_labels]

        print("prepare 1")
        for l in range(len(self.list_labels)):
            print(self.list_labels)
            for i in range(len(train_data)):
                if train_labels[i] == self.list_labels[l]:
                    self.group_train_data[l].append(train_data[i])
                    self.group_train_data_pos[l].append(i)
                    self.group_train_data_piss[l].append(self.train_data_piss[i])
                    self.group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])

        # Select shapelet for a group of label
        self.list_group_ppi = [[] for i in range(len(self.list_labels))]
        print("prepare 2")
        print(multiprocessing.cpu_count())
        pool = multiprocessing.Pool(processes=self.processes)
        print("prepare 3")
        if flag == 1:
            for l in range(len(self.list_labels)):
                for d in range(self.dim):
                    print("l:%s-%s" % (l,d))
                    # temp_ppi = pool.map(partial(self.find_ppi, l=l, d=d), range(len(self.group_train_data[l])))
                    temp_ppi = [self.find_ppi(i, l, d) for i in range(len(self.group_train_data[l]))]
                    list_ppi = []
                    for i in range(len(self.group_train_data[l])):
                        pii_in_i = temp_ppi[i]
                        for j in range(len(pii_in_i)):
                            list_ppi.append(pii_in_i[j])
                    list_ppi = np.asarray(list_ppi)
                    self.list_group_ppi[l].append(list_ppi)
        else:
            for l in range(len(self.list_labels)):
                for i in range(len(self.group_train_data[l])):
                    # temp_ppi = pool.map(partial(self.find_ppi2, l=l, i=i), range(self.dim))
                    temp_ppi = [self.find_ppi(i, l, d) for d in range(self.dim)]
                    self.list_group_ppi[l].append(temp_ppi)

        time2 = time.time() - time2
        print("window_size: %s - evaluating_time: %s" % (self.window_size, time2))

