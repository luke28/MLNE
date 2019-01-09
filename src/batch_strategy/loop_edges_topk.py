import numpy as np
import random


def batch_strategy(G, topk_params, rmapp, params, info):
    p = params
    def get_batch(maxx):
        now = 0
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
                batch_pos_weight[i] = 1.0 / info["q"][2]
                
                for j in xrange(p.num_sampled):
                    u = random.randint(0, len(rmapp) - 1)
                    v = random.randint(0, len(rmapp) - 1)
                    batch_w_neg[i * p.num_sampled + j] = u
                    batch_c_neg[i * p.num_sampled + j] = v
                    
                    ind = float(topk_params["in_degree"][v])
                    outd = float(topk_params["out_degree"][u])
                    
                    tmp = ind * outd / float(info["total_degree"]) / float(G.number_of_edges()) * float(len(rmapp) ** 2) / info["q"][2]
                    batch_neg_weight[i * p.num_sampled + j] = tmp
                now += 1
            yield batch_w_pos, batch_w_neg, batch_c_pos, batch_c_neg, batch_pos_weight, batch_neg_weight

    return get_batch
