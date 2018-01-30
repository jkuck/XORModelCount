import os
from sat_counter import SATCounter
from sparse_count import SparseCount
from sat import SAT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='examples/lang12.cnf')
parser.add_argument('--type', type=str, default='lbo')
parser.add_argument('--f', type=float, default=0.2)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--max_time', type=int, default=600)
parser.add_argument('--result_path', type=str, default='result/')
parser.add_argument('--confidence', type=float, default=0.95)
parser.add_argument('--regular', type=bool, default=False)
args = parser.parse_args()

problem = SAT(args.input, verbose=False)
if args.type == 'sc':
    counter = SparseCount(problem)
else:
    counter = SATCounter(problem, use_regular=args.regular)

if not os.path.isdir(args.result_path):
    os.mkdir(args.result_path)
if args.regular:
    type_str = 'regular'
else:
    type_str = 'original'
result_file = os.path.join(args.result_path, "%s_%s_%.3f_%s.txt" % (args.input.split('/')[-1], args.type, args.f, type_str))
if os.path.isfile(result_file):
    os.remove(result_file)

logger = open(result_file, "w")
logger.write("Problem name: %s\n" % args.input)
logger.write("n = %d, f = %f\n" % (problem.n, args.f))
logger.write("%s\n" % args.type)
logger.flush()
if args.type == 'ube':
    bound, m_star, time = counter.upper_bound_enumerate(f=args.f, min_confidence=args.confidence,
                                                        min_m=0, max_time=args.max_time)
    logger.write("Log bound: %f @ m = %d time = %f\n" % (bound, m_star, time))
elif args.type == 'ubo':
    bound, m_star, time, efficiency = counter.upper_bound(f=args.f, min_confidence=args.confidence,
                                                          max_time=args.max_time)
    logger.write("Log bound: %f @ m = %d time = %f efficiency = %f\n" % (bound, m_star, time, efficiency))
elif args.type == 'lbe':
    bound, m_star = counter.lower_bound_enumerate(f=args.f, min_confidence=args.confidence, max_time=args.max_time,
                                                  min_m=0, max_m=problem.n)
    logger.write("Log Bound: %f @ m = %d time = %f\n" % (bound, m_star, args.max_time))
elif args.type == 'lbo':
    bound, m_star, time, efficiency = counter.lower_bound(f=args.f, min_confidence=args.confidence,
                                                          max_time=args.max_time)
    logger.write("Log bound: %f @ m = %d time = %f efficiency = %f\n" % (bound, m_star, time, efficiency))
elif args.type == 'sc':
    bound = counter.sparse_count(f=args.f, min_confidence=args.confidence, max_time=args.max_time)
    logger.write("Log bound: %f\n" % bound)
else:
    print("Unrecognized type parameter, try again")
    exit(1)
logger.close()

