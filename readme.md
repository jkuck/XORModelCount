# Usage
** python main.py cnf_file_name problem_type f_value output_file_name max_time **

This command will solve the problem in cnf_file_name with given values for f and maximum time and write the results to output_file_name

List of possible problem types:
 * lbo: obtain a lower bound of the model count, using an exploration strategy that is fast, mostly reliable, but have no guarantees
 * lbe: obtain a lower bound of the model count, using an enumeration strategy that is slow but reliable
 * ubo: obtain an upper bound of the model count, using the exploration strategy
 * ube: obtain an upper bound of the model count, using the enumeration strategy

# File Description
 * main.py: wrapper that invokes the solver based on command line arguments
 * SATCounter.py: solver that implements the actual algorithm
 * SAT.py: SAT solver that invokes cryptominisat to solve SAT instances
 * timer.py: timer utility 
 * util.py: a set of support utilities, such as handling large numbers

# Dependencies
 * All source files are written for python2.7, no guarantees are given w.r.t. python3
 * Installation of the bigfloat package is required, see https://pypi.python.org/pypi/bigfloat/
 * Cryptominisat2 is authored by Mate Soos and can be downloaded at http://www.msoos.org/cryptominisat2/. Binaries for x64 are provided here for convenience.