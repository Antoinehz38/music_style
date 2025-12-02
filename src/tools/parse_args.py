import argparse

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run_from", type=str, default="new_conf")
    p.add_argument("--baseline", type=bool, default=False)

    p.add_argument("--vote", dest="vote", action="store_true")
    p.add_argument("--no-vote", dest="vote", action="store_false")
    p.set_defaults(vote=True)

    return p.parse_args()

