import argparse

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run_from", type=str, default="new_conf")

    return p.parse_args()