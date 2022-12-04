import os
import ray
import time

duration = int(os.environ["JOB_DURATION"])
value = int(os.environ["JOB_VALUE"])


@ray.remote
def f():
    for i in range(duration):
        print("job ", value, " step ", i)
        time.sleep(1)


ray.get(f.remote())
