#### Main file to import only one time the database
#### then keeping the data and running the algo-script
#### allowing fast re-running, avoiding long import time.


import sys
from imp import reload

from mnist import get_data
import class_algo

part1Cache = None

if __name__ == "__main__":
    while True:
        if not part1Cache:
            part1Cache = get_data()

        try:
            class_algo.main_classiffy(part1Cache)
        except Exception as e:
            print(e)

        print ("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()

        try:
            reload(class_algo)
        except Exception as e:
            print(e)


