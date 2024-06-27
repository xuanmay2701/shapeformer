def find_c_shapelet(list_ppi, c):
    sort_list_ppi = list_ppi[list_ppi[:, 3].argsort()]
    return sort_list_ppi[-c:]


def find_c_shapelet_non_overlab(list_ppi, c, p=0.1, p_inner=0.1, len_ts=100):
    sort_list_ppi = list_ppi[list_ppi[:, 3].argsort()]
    list_group_shapelet = [sort_list_ppi[-1]]
    list_pos = [len(sort_list_ppi)-1]
    for i in reversed(range(len(sort_list_ppi) - 1)):
        is_add = True
        s2 = int(sort_list_ppi[i, 1])
        e2 = int(sort_list_ppi[i, 2])
        d2 = int(sort_list_ppi[i, 5])
        # if e2 - s2 < int(0.0*len_ts) or e2 - s2 > int(0.2*len_ts):
        #     continue
        for j in range(len(list_group_shapelet)):
            s1 = int(list_group_shapelet[j][1])
            e1 = int(list_group_shapelet[j][2])
            d1 = int(list_group_shapelet[j][5])
            if d1 == d2:
                if s1 <= s2 and e1 >= e2:
                    if (e2-s2)/(e1-s1) > p_inner:
                        is_add = False
                        break
                elif s2 <= s1 and e2 >= e1:
                    if (e1-s1)/(e2-s2) > p_inner:
                        is_add = False
                        break
                elif s1 < s2 and e1 < e2:
                    if (e1 - s2) / min((e2 - s2), (e1 - s1)) > p:
                        is_add = False
                        break
                elif s1 > s2 and e1 > e2:
                    if (e2 - s1) / min((e2 - s2), (e1 - s1)) > p:
                        is_add = False
                        break
        if is_add:
            if len(list_group_shapelet) == c:
                break
            list_group_shapelet.append(sort_list_ppi[i])
            list_pos.append(i)

    if len(list_group_shapelet) < c:
        for i in reversed(range(len(sort_list_ppi) - 1)):
            if i not in list_pos:
                if len(list_group_shapelet) == c:
                    break
                list_group_shapelet.append(sort_list_ppi[i])

    return list_group_shapelet