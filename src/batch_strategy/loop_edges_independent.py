import numpy as np
import random


def batch_strategy(G, sub_params, topk_params, rmapp, params, info):
    p = params
    def get_batch(maxx):
        now = 0
        base_len = len(rmapp) - p.num_top
        edge_lst = [e for e in G.edges()]
        for _ in xrange(maxx):
            batch_w_pos = np.zeros(p.batch_size, dtype = np.int32)
            batch_w_neg = np.zeros(p.batch_size * p.num_sampled, dtype = np.int32)
            batch_c_pos = np.zeros(p.batch_size, dtype = np.int32)
            batch_c_neg = np.zeros(p.batch_size * p.num_sampled, dtype = np.int32)
            batch_pos_weight = np.zeros(p.batch_size, dtype = np.float32)
            batch_neg_weight = np.zeros(p.batch_size * p.num_sampled, dtype = np.float32)
            for i in xrange(p.batch_size):
                if now >= len(edge_lst):
                    now = 0
                u = rmapp[edge_lst[now][0]]
                v = rmapp[edge_lst[now][1]]
                batch_w_pos[i] = u
                batch_c_pos[i] = v
                if u >= base_len and v >= base_len:
                    batch_pos_weight[i] = 1.0  / info["q"][2]
                elif u >= base_len or v >= base_len:
                    batch_pos_weight[i] = 1.0 / info["q"][1]
                else:
                    batch_pos_weight[i] = 1.0 / info["q"][0]
                
                for j in xrange(p.num_sampled):
                    v = random.randint(0, len(rmapp) - 1)
                    u = random.randint(0, len(rmapp) - 1)
                    batch_w_neg[i * p.num_sampled + j] = u
                    batch_c_neg[i * p.num_sampled + j] = v
                    if u < base_len:
                        outd = float(sub_params["out_degree"][u])
                    else:
                        outd = float(topk_params["out_degree"][u - base_len])
                    if v < base_len:
                        ind = float(sub_params["in_degree"][v])
                    else:
                        ind = float(topk_params["in_degree"][v - base_len])
                    tmp = info["Z"][1] / info["Z"][0] * ind * outd / float(info["total_degree"])
                    if u >= base_len and v >= base_len:
                        batch_neg_weight[i * p.num_sampled + j] = tmp / info["q"][2]
                    elif u >= base_len or v >= base_len:
                        batch_neg_weight[i * p.num_sampled + j] = tmp / info["q"][1]
                    else:
                        batch_neg_weight[i * p.num_sampled + j] = tmp / info["q"][0]
                now += 1
            yield batch_w_pos, batch_w_neg, batch_c_pos, batch_c_neg, batch_pos_weight, batch_neg_weight

    return get_batch
