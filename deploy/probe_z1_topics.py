#!/usr/bin/env python3
import time
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

TOPICS = [
    "rt/api/z1/lowstate",
    "rt/z1/lowstate",
    "/z1/lowstate",
    "z1/lowstate",
    "rt/z1/state",
    "/z1/state",
    "z1/state",
]

got = {}

def make_handler(topic):
    def handler(msg: MotorStates_):
        got[topic] = True
        print(f"[OK] recv from {topic}")
        try:
            if hasattr(msg, "states") and len(msg.states) > 0:
                s0 = msg.states[0]
                print("  first motor state:", s0)
            else:
                print("  msg:", msg)
        except Exception as e:
            print("  print msg failed:", e)
    return handler

def main():
    ChannelFactoryInitialize(0, "eno1")

    subs = []
    for topic in TOPICS:
        try:
            sub = ChannelSubscriber(topic, MotorStates_)
            sub.Init(make_handler(topic), 1)
            subs.append(sub)
            print(f"[SUB] {topic}")
        except Exception as e:
            print(f"[FAIL] subscribe {topic}: {e}")

    print("Waiting 10 seconds ...")
    t0 = time.time()
    while time.time() - t0 < 10.0:
        time.sleep(0.1)

    print("\nSummary:")
    for topic in TOPICS:
        print(f"  {topic}: {'YES' if got.get(topic, False) else 'NO'}")

if __name__ == "__main__":
    main()