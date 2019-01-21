import numpy as np
import random
from alias_table_sampling import AliasTable

def batch_strategy(G, sub_params, topk_params, rmapp, params, info):
    p = params
    def get_batch(maxx):
        base_len = len(rmapp) - p.num_top
        edge_lst = [e for e in G.edges()]
        pos_edge_lst = [[] for _ in xrange(3)]
        pos_edge_weight = [1.0 / float(info["q"][i]) for i in xrange(3)]

        neg_node_lst = [[] for _ in xrange(2)]
        neg_ind_dis = [[] for _ in xrange(2)]
        neg_outd_dis = [[] for _ in xrange(2)]
        neg_edge_weight = [0.0 for _ in xrange(4)]
        for i in xrange(0, 3, 2):
            neg_edge_weight[i >> 1] = 1.0 / float(info["q"][i])
        for i in xrange(2, 4):
            neg_edge_weight[i] = 1.0 / float(info["q"][2])
        
        
        for e in  G.edges():
            u, v = e
            u = rmapp[u]
            v = rmapp[v]
            if u >= base_len and v >= base_len:
                pos_edge_lst[2].append((u, v))
            elif u >= base_len or v >= base_len:
                pos_edge_lst[1].append((u, v))
            else:
                pos_edge_lst[0].append((u, v))
        for i in xrange(3):
            if len(pos_edge_lst[i]) == 0:
                pos_edge_weight[i] = 0.0
        at_pos = AliasTable(pos_edge_weight)
        at_neg = AliasTable(neg_edge_weight)

        for u in G.nodes():
            u = rmapp[u]
            if u < base_len:
                neg_node_lst[0].append(u)
                neg_ind_dis[0].append(float(sub_params["in_degree"][u]))
                neg_outd_dis[0].append(float(sub_params["out_degree"][u]))
            else:
                neg_node_lst[1].append(u)
                neg_ind_dis[1].append(float(topk_params["in_degree"][u - base_len]))
                neg_outd_dis[1].append(float(topk_params["out_degree"][u - base_len]))
        at_ind = [AliasTable(neg_ind_dis[i]) for i in xrange(2)]
        at_outd = [AliasTable(neg_outd_dis[i]) for i in xrange(2)]
        for _ in xrange(maxx):
            batch_w_pos = np.zeros(p.batch_size, dtype = np.int32)
            batch_w_neg = np.zeros(p.batch_size * p.num_sampled, dtype = np.int32)
            batch_c_pos = np.zeros(p.batch_size, dtype = np.int32)
            batch_c_neg = np.zeros(p.batch_size * p.num_sampled, dtype = np.int32)
            for i in xrange(p.batch_size):
                idx_pos = at_pos.sample()
                idx = random.randint(0, len(pos_edge_lst[idx_pos]) - 1)
                batch_w_pos[i], batch_c_pos[i] = pos_edge_lst[idx_pos][idx]
                for j in xrange(p.num_sampled):
                    idx_neg = at_neg.sample()
                    if idx_neg == 0:
                        u_id = at_outd[0].sample()
                        u = neg_node_lst[0][u_id]
                        v_id = at_ind[0].sample()
                        v = neg_node_lst[0][v_id]
                    elif idx_neg == 1:
                        u_id = at_outd[1].sample()
                        u = neg_node_lst[1][u_id]
                        v_id = at_outd[1].sample()
                        v = neg_node_lst[1][v_id]
                    elif idx_neg == 2:
                        u_id = at_outd[0].sample()
                        u = neg_node_lst[0][u_id]
                        v_id = at_outd[1].sample()
                        v = neg_node_lst[1][v_id]
                    else:
                        u_id = at_outd[1].sample()
                        u = neg_node_lst[1][u_id]
                        v_id = at_outd[0].sample()
                        v = neg_node_lst[0][v_id]

                    batch_w_neg[i * p.num_sampled + j] = u
                    batch_c_neg[i * p.num_sampled + j] = v
            yield batch_w_pos, batch_w_neg, batch_c_pos, batch_c_neg

    return get_batch
