import random
import time


def random_short_wait(min_seconds=1, max_seconds=2):
    """sleep a random short interval"""
    time.sleep(random.randint(min_seconds, max_seconds))


def random_long_wait(min_seconds=5, max_seconds=30):
    """sleep a random long interval"""
    time.sleep(random.randint(min_seconds, max_seconds))
