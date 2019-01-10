import numpy as np
import os

from utils import common_tools as ct

def params_handler(params, info, pre_res, **kwargs):
    ct.check_attr(params, "times", 100)
    return {}

@ct.module_decorator
def loop(params, info, pre_res, **kwargs):
    res = params_handler(params, info, pre_res)

    for num in xrange(params["times"]):
        info["log"].info("[+] The %d-th loop start!" % num)
        print("[+] The %d-th loop start!" % num)
        for module in params["loop_modules"]:
            mdl_name = module["func"]
            mdl_params = module["params"]
            mdl = __import__("%s.%s" % (mdl_name, mdl_params["func"]), fromlist = [mdl_name])
            res[mdl_name] = getattr(mdl, mdl_name)(mdl_params,
                info = info,
                pre_res = res,
                mdl_name = mdl_name) 
    return res