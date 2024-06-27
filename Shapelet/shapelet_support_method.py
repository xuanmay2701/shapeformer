import numpy as np


def entropy(num_of_pos, num_of_neg):
    total_len = num_of_pos + num_of_neg
    result = 0
    if num_of_pos / total_len > 0: result -= (num_of_pos / total_len) * np.log2(num_of_pos / total_len)
    if num_of_neg / total_len > 0: result -= (num_of_neg / total_len) * np.log2(num_of_neg / total_len)
    return result


def binary_inforgain(g1_num_of_pos, g1_num_of_neg, g2_num_of_pos, g2_num_of_neg):
    total_len = g1_num_of_pos + g1_num_of_neg + g2_num_of_pos + g2_num_of_neg
    g1_len = g1_num_of_pos + g1_num_of_neg
    g2_len = g2_num_of_pos + g2_num_of_neg
    init_num_of_pos = g1_num_of_pos + g2_num_of_pos
    init_num_of_neg = g1_num_of_neg + g2_num_of_neg

    return entropy(init_num_of_pos, init_num_of_neg)\
           -(g1_len/total_len)*entropy(g1_num_of_pos, g1_num_of_neg)\
           -(g2_len/total_len)*entropy(g2_num_of_pos, g2_num_of_neg)


def sort_by_sub_dist(list_sub_dist, input_label):
    # Sort list_table by list_sub_dist
    list_sd_table = np.array([input_label, list_sub_dist]).transpose()
    list_sd_table = list_sd_table[list_sd_table[:, 1].argsort()]

    list_label_table = list_sd_table[:, 0]
    list_subdist_table = list_sd_table[:, 1]

    return list_label_table, list_subdist_table


def find_best_split_point_and_info_gain(list_sub_dist, input_label, target_class):

    list_label_table, list_subdist_table = sort_by_sub_dist(list_sub_dist=list_sub_dist,
                                                            input_label=input_label)

    best_ig = -1
    g1_pos = 0
    g1_neg = 0
    g2_pos = 0
    g2_neg = 0
    for i in list_label_table:
        if i == target_class:
            g2_pos += 1
        else:
            g2_neg += 1

    t_pos = g2_pos

    rate = 0.2
    for i in range(len(list_label_table)-1):
        if list_label_table[i] == target_class:
            g1_pos += 1
            g2_pos -= 1
        else:
            g1_neg += 1
            g2_neg -= 1
        if round(g1_pos/t_pos,1) == rate:
            inforgain = binary_inforgain(g1_num_of_pos=g1_pos, g1_num_of_neg=g1_neg,
                                          g2_num_of_pos=g2_pos, g2_num_of_neg=g2_neg)
            best_ig = max(best_ig, inforgain)
            if rate < 1:
                rate += 0.2
            else:
                break

    return best_ig