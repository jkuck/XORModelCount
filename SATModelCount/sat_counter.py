import sys
import time
from scipy.special import binom as binom
from timer import Timer
from util import *
import numpy as np

class SATCounter:
    def __init__(self, sat_problem, verbose=True, use_regular=False):
        """ Each SATCounter solves a specific sat problem that must be specified at creation """
        self.sat = sat_problem
        self.verbose = verbose
        self.binom = BigBinom(sat_problem.n)
        self.use_regular = use_regular

    def lower_bound(self, f, min_confidence=0.9, max_time=600, best_slack=0.1):
        """ Get a log lower bound for the SAT problem with specified constraint density and minimum confidence
            Set the best_slack if we are satisfied with a solution that is 1 / (1 + best_slack) smaller than best
            try, and we wish the algorithm returns early before max_time
            Returns a tuple of the best bound obtained, the best m to give us that bound, and the percentage of trials
            performed on the best m. If no results are obtained returns -1"""
        # Start a timer
        timer = Timer(max_time)

        # Define some shorthands
        ln = math.log
        sqrt = math.sqrt
        delta = 1 - min_confidence
        n = self.sat.n
        nan = float('NaN')
        big_two = BigFloat(2.0)
        big_zero = BigFloat(0.0)

        # Define required variables
        trial_count = [0] * n            # The number of trials performed on m
        true_count = [0] * n             # The number of trials m is satisfiable
        false_count = [0] * n            # The number of trials m is unsatisfiable
        incentive = [0.0] * n            # The incentive to perform trial on m in the next step
        trial_time = [0.0] * n           # The amount of time spent on the investigation of a certain m
        total_run = 0                    # The total number of of trials performed on all m

        while True:
            # Select the m with the strongest incentive to investigate
            if self.verbose:
                print("Finding the best m to explore  ....."),
            start_time = time.time()

            max_exp = BigFloat(-1)
            max_exp_m = 0

            # Compute the unnormalized incentive for all values of m
            for m in range(0, n):
                if trial_count[m] != 0:
                    c = float(true_count[m]) / trial_count[m]

                    # The expected # of trials we can perform if the current m is the bandit we will explore
                    expected_trial = timer.time() / (trial_time[m] / trial_count[m]) / 2 + trial_count[m]
                    if expected_trial * c > -ln(delta):     # We have enough time to obtain some kind of bound
                        kappa = -3 * ln(delta) + sqrt(ln(delta) ** 2 - 8 * c * expected_trial * ln(delta))
                        kappa /= 2 * (c * expected_trial + ln(delta))
                        if c != 0:
                            incentive[m] = (big_two ** m) * c / (1 + kappa)
                        else:
                            incentive[m] = big_zero
                        if incentive[m] < 0:
                            incentive[m] = big_zero
                    else:
                        incentive[m] = big_zero

                    # Find m with best outlook
                    if incentive[m] > max_exp:
                            max_exp = incentive[m]
                            max_exp_m = m

            # Limit our range of exploration between [max_exp_m - 8, max_exp_m + 8)
            m_min = max_exp_m - 8
            if m_min < 0:
                m_min = 0
            m_max = max_exp_m + 8
            if m_max > n:
                m_max = n

            # Normalize incentive to [0, 1] and add exploration term
            for m in range(m_min, m_max):
                if trial_count[m] == 0:
                    incentive[m] = 100.0    # Very strong incentive if this has never been explored before
                else:
                    if max_exp == 0:
                        incentive[m] = 0.0
                    else:
                        incentive[m] = float(incentive[m] / max_exp)
                    incentive[m] += math.sqrt(2.0 * math.log(total_run) / trial_count[m])

            # Find the m with best incentive as m_star
            m_star = int(np.argmax(incentive[m_min:m_max]) + m_min)
            incentive_star = np.max(incentive[m_min:m_max])

            if self.verbose:
                print("Found m = %d with incentive %f. center = %d, time left %f" % (m_star, incentive_star, max_exp_m, timer.time()))

            # Perform a trial on that m
            if self.use_regular:
                self.sat.add_regular_constraints(m_star, f)
            else:
                self.sat.add_parity_constraints(m_star, f)
            outcome = self.sat.solve(max_time=timer.time())
            if outcome is True:
                true_count[m_star] += 1
                trial_count[m_star] += 1
            elif outcome is False:
                false_count[m_star] += 1
                trial_count[m_star] += 1

            if self.verbose:
                status_msg = "%d/%d satisfiable @ m = %d" % (true_count[m_star], trial_count[m_star], m_star)
                if outcome is True:
                    print("SAT......" + status_msg)
                elif outcome is False:
                    print("UNSAT...." + status_msg)
                else:
                    print("timeout.." + status_msg)

            # Check if that m_star satisfies our termination criteria
            if trial_count[m_star] != 0:
                c = float(true_count[m_star]) / trial_count[m_star]
                confidence = 1 - math.exp(-best_slack * best_slack * c * trial_count[m_star] /
                                          ((1 + best_slack) * (2 + best_slack)))
                if confidence >= min_confidence:
                    best_log_bound = m_star * ln(2) + ln(c) - ln(1 + best_slack)
                    if self.verbose:
                        print("Complete! best log bound is %f @ m_star = %d" % (best_log_bound, m_star))
                        print("%d/%d satisfiable, kappa = %f" % (true_count[m_star], trial_count[m_star], best_slack))
                    break

            end_time = time.time()
            trial_time[m_star] += end_time - start_time

            # If timer times out, find out the best bound we have
            if timer.timeout():
                best_log_bound = nan
                m_star = -1
                kappa_star = 0.0
                for m in range(0, n):
                    if trial_count[m] == 0:
                        continue
                    c = float(true_count[m]) / trial_count[m]
                    T = trial_count[m]
                    if c * T > -ln(delta):
                        kappa = -3 * ln(delta) + sqrt(ln(delta) ** 2 - 8 * c * T * ln(delta))
                        kappa /= 2 * (c * T + ln(delta))
                        cur_bound = m * ln(2) + ln(c) - ln(1 + kappa)
                        if math.isnan(best_log_bound) or cur_bound > best_log_bound:
                            best_log_bound = cur_bound
                            m_star = m
                            kappa_star = kappa
                if self.verbose:
                    print("Time up! best log bound is %f @ m_star = %d" % (best_log_bound, m_star))
                    print("%d/%d satisfiable, kappa = %f" % (true_count[m_star], trial_count[m_star], kappa_star))
                break

            total_run += 1

        if self.verbose:
            print("Time usage: %fs, efficiency: %f" % (max_time - timer.time(), float(trial_count[m_star]) / total_run))
        return best_log_bound, m_star, max_time - timer.time(), float(trial_count[m_star]) / total_run

    def lower_bound_enumerate(self, f, min_confidence=0.9, max_time=600, min_m=0, max_m=-1):
        """ Get a lower bound for the SAT problem with specified constraint density and minimum confidence
        This does a brute force search on all possible m, each possible m is investigated until max_time
        Returns a tuple of the log of best bound obtained, the best m to give us that bound"""
        # Define some shorthands
        ln = math.log
        sqrt = math.sqrt
        delta = 1 - min_confidence
        n = self.sat.n
        nan = float('NaN')

        if max_m < 0:
            max_m = n

        log_bound_list = []
        for i in range(0, n):
            log_bound_list.append(nan)

        for m in range(min_m, max_m):
            # Start a timer
            timer = Timer(max_time)

            # Define required variables
            true_count = 0             # The number of trials m is satisfiable
            false_count = 0            # The number of trials m is unsatisfiable

            if self.verbose:
                print("Performing trials on m = " + str(m))

            while True:
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
                    if true_count + false_count == 0:
                        log_bound_list[m] = nan
                        if self.verbose:
                            print("m = %d failed, not a single instance can be evaluated" % m)
                        break

                    T = float(true_count + false_count)
                    c = float(true_count) / T
                    if c * T > -ln(delta):
                        kappa = -3 * ln(delta) + sqrt(ln(delta) ** 2 - 8 * c * T * ln(delta))
                        kappa = kappa / 2 / (c * T + ln(delta))
                        log_bound_list[m] = m * ln(2) + ln(c) - ln(1 + kappa)
                        if self.verbose:
                            print("m = %d complete, obtained log bound = %f" % (m, log_bound_list[m]))
                            print("%d/%d satisfiable, kappa = %f" % (true_count, true_count + false_count, kappa))
                    else:
                        log_bound_list[m] = nan
                        if self.verbose:
                            print("m = %d failed, too few instance evaluated" % m)
                    break

        # Find the best bound obtained
        m_star = -1
        best_bound = nan
        for m in range(min_m, max_m):
            if math.isnan(best_bound) or (not math.isnan(log_bound_list[m]) and log_bound_list[m] > best_bound):
                best_bound = log_bound_list[m]
                m_star = m
        if self.verbose:
            print("Complete! best log bound is " + str(best_bound) + " @ " + str(m_star))
        return best_bound, m_star

    def upper_bound(self, f, min_confidence=0.9, bold=False, max_time=600):
        """ Get a upper bound for the SAT problem with specified constraint density and minimum confidence
        Returns a tuple of the best bound obtained, the best m to give us that bound, time usage,
        percentage of computational resources used on optimal m. If no results are obtained returns -1"""

        timer = Timer(max_time)
        start_time = time.time()

        # Define some notation shorthands
        ln = math.log
        delta = 1 - min_confidence
        n = self.sat.n
        T = int(math.ceil(24 * ln(1 / delta)))
        if self.verbose:
            print("Requires %d samples to verify" % T)

        # If in bold mode, estimate the place where the problem becomes UNSAT and limit our search in that region
        max_m_global = n
        if bold:
            if self.verbose:
                print("Searching for the maximum reasonable m")
            for m in range(0, n, 5):
                self.sat.add_parity_constraints(m, f)
                outcome = self.sat.solve(timer.time())
                if self.verbose:
                    print("[m=%d, %s]" % (m, str(outcome))),
                    if outcome is not True:
                        print("")
                if outcome is None:
                    return None
                if outcome is False:
                    max_m_global = m + 5
                    break
                if timer.timeout():
                    print("Timeout!")
                    return float('nan'), -1, max_time, 0

        if self.verbose:
            print("Starting the search maximum m of %d " % max_m_global)
            print("Computing expected bound if trial successful")

        # Upper bound that would be obtained if half the bins are empty
        upper_bound_list = []
        for m in range(0, max_m_global + 1):
            expected_bound = self.upper_bound_expected(m, f)
            upper_bound_list.append(expected_bound)
            if self.verbose:
                print("[%d:%.2f]" % (m, float(bf.log(expected_bound)))),
                if m == max_m_global:
                    print("")

        # Define required variables
        trial_count = [0] * n                   # The number of trials performed on m
        true_count = [0] * n                    # The number of trials m is satisfiable
        false_count = [0] * n                   # The number of trials m is unsatisfiable
        incentive = [BigFloat(0)] * n           # The incentive to perform trial on m in the next step
        posterior_success_prob = [0.0] * n
        total_run = 0

        while True:
            if self.verbose:
                print("Finding the best m to explore  ....."),
            # Select the m with the strongest incentive to investigate
            max_exp = BigFloat(-1)
            max_exp_m = 0
            # Compute the un-normalized incentive for all values of m
            for m in range(max_m_global - 1, -1, -1):
                if true_count[m] < T / 2 and trial_count[m] != 0:
                    incentive[m] = posterior_success_prob[m] / upper_bound_list[m] / m
                else:
                    incentive[m] = BigFloat(0)
                if incentive[m] > max_exp:
                    max_exp = incentive[m]
                    max_exp_m = m

            # Limit our range of exploration between [max_exp_m - 5, max_exp_m + 5)
            m_min = max_exp_m - 5
            if m_min < 0:
                m_min = 0
            m_max = max_exp_m + 5
            if m_max > max_m_global:
                m_max = max_m_global

            # Normalize incentive between [0, 1] and add exploration term
            for m in range(m_min, m_max):
                if trial_count[m] == 0:
                    incentive[m] = BigFloat(100)        # Very strong incentive if this has never been explored before
                else:
                    if max_exp == BigFloat(0):
                        incentive[m] = BigFloat(0)
                    else:
                        incentive[m] = incentive[m] / max_exp
                    incentive[m] += math.sqrt(2.0 * math.log(total_run) / trial_count[m])

            # Find the m_star that maximizes incentive
            incentive_star = BigFloat(-1)
            m_star = -1
            for m in range(m_min, m_max):
                if incentive[m] > incentive_star:
                    incentive_star = incentive[m]
                    m_star = m
            if self.verbose:
                print("Found m = %d with incentive %f. center = %d" % (m_star, incentive_star, max_exp_m))

            # Perform a trial on that m
            self.sat.add_parity_constraints(m_star, f)
            outcome = self.sat.solve(timer.time())
            if timer.timeout():
                print("Timeout!")
                return float('nan'), -1, max_time, 0

            if outcome is True:
                true_count[m_star] += 1
                trial_count[m_star] += 1
            elif outcome is False:
                false_count[m_star] += 1
                trial_count[m_star] += 1

            if self.verbose:
                status_msg = "%d/%d satisfiable @ m = %d" % (true_count[m_star], trial_count[m_star], m_star)
                if outcome is True:
                    print("SAT......" + status_msg)
                elif outcome is False:
                    print("UNSAT...." + status_msg)
                else:
                    print("timeout.." + status_msg)

            # Update posterior success probability
            if trial_count[m_star] > 0:
                posterior_success_prob[m_star] = self.posterior_success_prob(T, trial_count[m_star], true_count[m_star])

            # Check if that m_star satisfies our termination criteria
            if trial_count[m_star] >= T and true_count[m_star] < false_count[m_star]:
                best_bound = upper_bound_list[m_star]
                if self.verbose:
                    print("Complete! best log bound is %f @ m_star = %d" % (float(bf.log(best_bound)), m_star))
                break

            total_run += 1

        end_time = time.time()
        if self.verbose:
            print("Time usage: %fs, efficiency: %f" % (end_time - start_time, float(T) / total_run))
        return float(bf.log(best_bound)), m_star, end_time - start_time, float(T) / total_run

    def upper_bound_enumerate(self, f, min_confidence=0.9, min_m=0, max_m=-1, max_time=600):
        """ Get a log upper bound for the SAT problem with specified constraint density and minimum confidence
        This does a brute force search on all possible m, each possible m is investigated for adequate number of times
        Returns the best bound, m to give us that bound, and time consumed on that particular m """
        # Define some notation shorthands
        delta = 1 - min_confidence
        n = self.sat.n
        if max_m < 0:
            max_m = n
        T = int(round(24 * math.log(1 / delta)))
        if self.verbose:
            print("Requires %d samples to verify" % T)

        # Iterate from small m to large, smaller m gives better bounds
        for m in range(min_m, max_m):
            timer = Timer(max_time)
            time_out = False

            print("Inspecting m = %d" % m)
            start_time = time.time()
            true_count = 0              # The number of trials m is satisfiable
            false_count = 0             # The number of trials m is unsatisfiable
            for trial in range(0, T):
                self.sat.parityConstraints(m, f)
                outcome = self.sat.solve(timer.time())
                if outcome is True:
                    true_count += 1
                    print("T"),
                elif outcome is False:
                    false_count += 1
                    print("F"),
                else:
                    print("E"),
                    time_out = True
                    break
                if timer.timeout():
                    time_out = True
                    break

            end_time = time.time()
            print("\n %d out of %d evaluated to be satisfiable" % (true_count, T))
            if true_count < false_count and not time_out:
                actual_bound = self.upper_bound_expected(m, f)
                print("Solved @ m = %d with log upper bound %f" % (m, float(bf.log(actual_bound))))
                return float(bf.log(actual_bound)), m, end_time - start_time
        return -1, -1, -1

    @staticmethod
    def posterior_success_prob(T, k, l):
        """ If out of the first k samples, l are one, this computes the MLE of the probability
        that less than half of the T samples shall be 1 (Generally T should not be greater than 100
        to guarantee numerical stability """
        # MLE for probability a sample is 1
        p = float(l) / k
        sum = 0.0
        for i in range(0, int(T / 2 - l)):
            sum += binom(T - k, i) * (p ** i) * ((1 - p) ** (T - k - i))
        return sum

    def upper_bound_expected(self, m, f, tolerance=0.001):
        """ Obtain the expected upper bound for set size given m, and f, accurate to given tolerance """
        # Shorthand definition
        two_to_m = BigFloat(2.0) ** m

        # Use binary search to find the minimum q so that z > 3/4
        q_min = BigFloat(1.0)
        q_max = BigFloat(2.0) ** self.sat.n
        for iteration in range(0, self.sat.n + 10):
            q_mid = bf.sqrt(q_min * q_max)  # Search by geometric mean
            v = q_mid / two_to_m * (1 + self.compute_eps_q(m, q_mid, f) - q_mid / two_to_m)
            z = 1 - v / (v + (q_mid / two_to_m) ** 2)
            if z > 3.0 / 4:
                q_max = q_mid
            else:
                q_min = q_mid

            # If difference between q_min and q_max is less than tolerance, stop the search
            if q_max < q_min * (1 + tolerance):
                break
        return bf.sqrt(q_max * q_min)

    def compute_eps_q(self, m, q, f):
        """ This function computes epsilon(n, m, q, f) * (q - 1), and is optimized for multiple queries """
        # Use binary search to find maximum w_star so that sum_{w = 1}^{w_star} C(n, w) <= q - 1
        # The possible w_star lies in [w_min, w_max]
        w_min = 0
        w_max = self.sat.n
        while w_min != w_max:
            w = int(math.ceil(float(w_min + w_max) / 2))    # If w_min + 1 = w_max, assign w to w_max
            if self.binom.binom_sum(w) < q:
                w_min = w
            elif self.binom.binom_sum(w) == q:
                w_min = w
                break
            else:
                w_max = w - 1
        w_star = w_min
        r = q - 1 - self.binom.binom_sum(w_star)

        # Compute eps * (q - 1)
        epsq = r * (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** (w_star + 1))) ** m
        for w in range(1, w_star + 1):
            epsq += self.binom.binom(w) * (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** w)) ** m
        return epsq

    @staticmethod
    def compute_eps_q_static(n, m, q, f, binom=None):
        """ This function computes epsilon(n, m, q, f) * (q - 1), and is not optimized for multiple queries """
        # Use binary search to find maximum w_star so that sum_{w = 1}^{w_star} C(n, w) <= q - 1
        # The possible w_star lies in [w_min, w_max]
        w_min = 0
        w_max = n
        if binom is None:
            binom = BigBinom(n)
        while w_min != w_max:
            w = int(math.ceil(float(w_min + w_max) / 2))    # If w_min + 1 = w_max, assign w to w_max
            if binom.binom_sum(w) < q:
                w_min = w
            elif binom.binom_sum(w) == q:
                w_min = w
                break
            else:
                w_max = w - 1
        w_star = w_min
        r = q - 1 - binom.binom_sum(w_star)

        # Compute eps * (q - 1)
        epsq = r * (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** (w_star + 1))) ** m
        for w in range(1, w_star + 1):
            epsq += binom.binom(w) * (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** w)) ** m
        return epsq

    def compute_f_star(self, m):
        """Compute the minimum f to guarantee a constant factor approximation"""
        q = BigFloat(2.0) ** (m + 2)
        threshold = 31 / 5

        # Find f_star by binary search
        f_left = 0.0
        f_right = 0.5
        for iteration in range(0, 10):
            f_mid = (f_left + f_right) / 2
            # Compute eps from f
            eps = self.compute_eps_q(m, q, f_mid)
            if eps < threshold:
                f_right = f_mid
            else:
                f_left = f_mid
        return (f_left + f_right) / 2

    @staticmethod
    def compute_f_star_static(n, m):
        """Compute the minimum f to guarantee a constant factor approximation"""
        q = BigFloat(2.0) ** (m + 2)
        threshold = 8

        binom = BigBinom(n)
        # Find f_star by binary search
        f_left = 0.0
        f_right = 0.5
        for iteration in range(0, 10):
            f_mid = (f_left + f_right) / 2
            # Compute eps from f
            eps = SATCounter.compute_eps_q_static(n, m, q, f_mid, binom)
            if eps < threshold:
                f_right = f_mid
            else:
                f_left = f_mid
        return (f_left + f_right) / 2

