__author__ = 'shengjia'

from os import sys
from SATCounter import *
from SAT import SAT

# Usage
# param1: name of the input problem file
# param2: type of test we want: lbo, lbe, ubo, ube
# param3: f if WISH, eps if ApproxMC
# param4: max time
if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Incorrect parameters. Usage: instance_name, problem_type, f_value, output_file_name, max_time(optional)")
        exit(1)

    problem_name = sys.argv[1]
    problem = SAT(problem_name, verbose=False)
    counter = SATCounter(problem)

    ofstream = open(sys.argv[4], "w")

    # Run parameters
    if sys.argv[2] == "lbo":
        type = "lower bound optimal exploration"
    elif sys.argv[2] == "lbe":
        type = "lower bound enumerate"
    elif sys.argv[2] == "ubo":
        type = "upper bound optimal exploration"
    elif sys.argv[2] == "ube":
        type = "upper bound enumerate"
    else:
        print("Unrecognized type parameter, try again")
        exit(1)

    f = float(sys.argv[3])
    max_time = float(sys.argv[5])
    min_confidence = 0.95

    ofstream.write("Problem name: " + sys.argv[1] + "\n")
    ofstream.write("n = " + str(problem.n) + ", f = " + str(f) + " \n")
    ofstream.write(type + "\n")
    if type == "upper bound enumerate":
        bound, m_star, time = counter.upperBoundEnumerate(f=f, min_confidence=min_confidence, min_m=0)
        ofstream.write("Log bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(time))
    elif type == "upper bound optimal exploration":
        bound, m_star, time, efficiency = counter.upperBound(f=f, min_confidence=min_confidence)
        ofstream.write("Log bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(time) +
                       ", efficiency = " + str(efficiency))
    elif type == "lower bound enumerate":
        bound, m_star = counter.lowerBoundEnumerate(f=f, min_confidence=min_confidence, max_time=max_time,
                                                    min_m=0, max_m=problem.n)
        ofstream.write("Log Bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(max_time))
    elif type == "lower bound optimal exploration":
        bound, m_star, time, efficiency = counter.lowerBound(f=f, min_confidence=min_confidence, max_time=max_time)
        ofstream.write("Log Bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(time) +
                       ", efficiency = " + str(efficiency))
    ofstream.close()
