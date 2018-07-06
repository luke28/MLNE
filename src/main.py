import os
import sys
import json
import argparse
import numpy as np
import networkx as nx
import random
from operator import itemgetter

from utils.env import *
#from utils.datahandler import DataHandler as dh
from utils import common_tools as ct


def init(args, params, whole_params):
    info = {}
    info["time"] = ct.get_time_str()
    info["whole_params"] = whole_params
    info["conf_name"] = args.conf
    info["res_home"] = os.path.join(os.path.join(RES_PATH, args.conf), info["time"])
    #info["data_path"] = DATA_PATH
    info["network_path"] = os.path.join(DATA_PATH, params["network_file"])
    info["num_nodes"] = params["num_nodes"]
    # if not exists, then mkdir
    ct.mkdir(info["res_home"])

    log_path = os.path.join(LOG_PATH, info["time"] + ".log")
    info["log"] = ct.get_logger(log_path)
    ct.symlink(log_path, os.path.join(LOG_PATH, "new_log"))
    ct.symlink(info["res_home"], os.path.join(RES_PATH, "new_res"))
    return info

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--conf', type = str, default = "test")
    args = parser.parse_args()
    params = ct.load_json_file(os.path.join(CONF_PATH, args.conf + ".json"))
    info = init(args, params["static_info"], params)

    res = {}
    for module in params["run_modules"]:
        mdl_name = module["func"]
        mdl_params = module["params"]
        mdl = __import__(mdl_name + '.' + mdl_params["func"], fromlist = [mdl_name])
        res[mdl_name] = getattr(mdl, mdl_name)(mdl_params, info = info, res = res, mdl_name = mdl_name)
        print res

if __name__ == "__main__":
    main()
