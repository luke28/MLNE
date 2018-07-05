import os
import sys
import json
import argparse
import numpy as np
import networkx as nx
import random
from operator import itemgetter



def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--conf', type = str, default = "test")
    args = parser.parse_args()
    static_info = init(params["static_info"])
