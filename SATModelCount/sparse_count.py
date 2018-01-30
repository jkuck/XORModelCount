import sys
import time
from scipy.special import binom as binom
from timer import Timer
from util import *
import numpy as np


class SparseCount:
    def __init__(self, sat_problem, verbose=True):
        """ Each SATCounter solves a specific sat problem that must be specified at creation """
        self.sat = sat_problem
        self.verbose = verbose
        self.binom = BigBinom(sat_problem.n)

    def sparse_count(self, f, min_confidence=0.9, max_time=600):
        """ Get a lower bound for the SAT problem with specified constraint density and minimum confidence
        This does a brute force search on all possible m, each possible m is investigated until max_time
        Returns a tuple of the log of best bound obtained, the best m to give us that bound"""
        # Define some shorthands
        ln = math.log
        delta = 1 - min_confidence
        n = self.sat.n

        # Start a timer
        timer = Timer(max_time)
        time_out = False

        alpha = 0.0042
        T = int(math.ceil(-ln(delta) / alpha * ln(n)))

        cumulative_sum = BigFloat(0.0)
        for m in range(0, n):
            # Define required variables
            true_count = 0             # The number of trials m is satisfiable
            false_count = 0            # The number of trials m is unsatisfiable

            if self.verbose:
                print("Performing trials on m = " + str(m))

            for t in range(T):
                # Perform a trial on that m
                self.sat.add_parity_constraints(m, f)
                outcome = self.sat.solve(timer.time())
                if outcome is True:
                    true_count += 1
                elif outcome is False:
                    false_count += 1

                if self.verbose:
                    status_msg = "%d/%d satisfiable @ m = %d" % (true_count, true_count + false_count, m)
                    if outcome is True:
                        print("SAT......" + status_msg)
                    elif outcome is False:
                        print("UNSAT...." + status_msg)
                    else:
                        print("timeout.." + status_msg)

                # If timer times out, find out the best bound we have
                if timer.timeout():
                    if self.verbose:
                        print("Timeout")
                    time_out = True
                    break

            if time_out:
                break

            if true_count > false_count:
                cumulative_sum += BigFloat(2.0) ** max(m-1, 0)
                print("Satisfiable @ m = %d" % m)
            else:
                break

        if time_out:
            return None
        else:
            return float(bf.log(cumulative_sum))


# Plot the difficulty of running lang12 w.r.t. constraint density