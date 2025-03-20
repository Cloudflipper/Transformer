import parse
import numpy as np
import transformer

def train():
    priviledge, obs_with_noise = parse.parse_data()
    