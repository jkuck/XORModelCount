import os
from sat_counter import SATCounter
from sat import SAT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='examples/wff.3.100.150.cnf')
parser.add_argument('--type', type=str, default='lbo')
parser.add_argument('--f', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--max_time', type=int, default=3600)
parser.add_argument('--result_path', type=str, default='result/')
parser.add_argument('--confidence', type=float, default=0.95)
args = parser.parse_args()

problem = SAT(args.input, verbose=False)
counter = SATCounter(problem)

if not os.path.isdir(args.result_path):
    os.mkdir(args.result_path)
result_file = os.path.join(args.result_path, "%s_%s.txt" % (args.input.split('/')[-1], args.type))
if os.path.isfile(result_file):
    os.remove(result_file)

logger = open(result_file, "w")
logger.write("Problem name: " + args.input + "\n")
logger.write("n = " + str(problem.n) + ", f = " + args.f + " \n")
logger.write("s\n" % args.type)
if args.type == 'ube':
    bound, m_star, time = counter.upperBoundEnumerate(f=args.f, min_confidence=args.min_confidence,
                                                      min_m=0, max_time=args.max_time)
    logger.write("Log bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(time))
elif args.type == 'ubo':
    bound, m_star, time, efficiency = counter.upperBound(f=args.f, min_confidence=args.min_confidence,
                                                         max_time=args.max_time)
    logger.write("Log bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(time) +
                   ", efficiency = " + str(efficiency))
elif args.type == 'lbe':
    bound, m_star = counter.lowerBoundEnumerate(f=args.f, min_confidence=args.min_confidence, max_time=args.max_time,
                                                min_m=0, max_m=problem.n)
    logger.write("Log Bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(args.max_time))
elif args.type == 'lbo':
    bound, m_star, time, efficiency = counter.lowerBound(f=args.f, min_confidence=args.min_confidence,
                                                         max_time=args.max_time)
    logger.write("Log Bound: " + str(bound) + " @ m = " + str(m_star) + ", time = " + str(time) +
                   ", efficiency = " + str(efficiency))
else:
    print("Unrecognized type parameter, try again")
    exit(1)
logger.close()

