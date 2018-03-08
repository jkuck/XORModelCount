from __future__ import division
from scipy.special import binom as binom
from timer import Timer
from bigfloat import BigFloat
import bigfloat as bf
import random
import math
import numpy as np
import operator as op
import time
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from plot_runtimeVSbound import read_files_moreInfo, log_2_Z
import sys
sys.path.insert(0, '/Users/jkuck/research/winter_2018/low_density_parity_checks/notes')

try:
    import cPickle as pickle
except ImportError:
    import pickle

from var_computation import log_g, get_Ax_zero_log_probs, get_Ax_zero_log_probs_all
from var_computation import get_Ax_zero_probs

##no need to load cach with Tri's new super fast implementation
##try:
##    with open('.log_g_cache', 'rb') as cache_file:
##        log_g.cache = pickle.load(cache_file)
##except IOError:
##    pass

EXPERIMENT_FOLDER = 'heatmap_result_fireworksTIMEOUTcomplete'
PROBLEM_NAME = 'tire-1'
REPEATS = 10
FILE_COUNT = 10
PLOT_ACTUAL_POINTS = False
ANNOTATE_PLOTS = False
############ experiment parameters ############
# 'f_block': 1 or '1minusF', the probability with which elements in the blocks of size k are set to 1
# 'permute': True or False, whether the columns of A were permuted
# 'k': integer specifying the block size, 'maxConstant' specifying the largest constant floor(n/m), None specifying 
#      some blocks are size floor(n/m) and some are ceiling(n/m) to fill all columns
# 'allOnesConstraint': True or False, whether a parity constraint containing all variables was included

all_params_to_plot = [
     {'f_block': 1, #original baseline method where all elements are iid
    'permute': False,
    'k': 0,
    'allOnesConstraint': False,},
    
    {'f_block': '1minusF',#kMaxConst_params = 
     'permute': True,
     'k': None,#'maxConstant',
     'allOnesConstraint': False,},

    {'f_block': '1minusF', #k3_params =
     'permute': True,
     'k': 3,
     'allOnesConstraint': False,}]

def get_filebase(params):
    filebase = '%s/%s/f_block=%s_permute=%s_k=%s_allOnesConstraint=%s_REPEATS=%d_expIdx=' \
        % (EXPERIMENT_FOLDER, PROBLEM_NAME, params['f_block'], params['permute'], \
           params['k'], params['allOnesConstraint'], REPEATS)
    return filebase

def get_n(problem_name):
    problem_filename = "/Users/jkuck/research/winter_2018/low_density_parity_checks/SAT_problems_cnf/%s.cnf" % problem_name

    f = open(problem_filename, 'r')
    for line in f:
        if line[0] == 'p': #parameters line
            params = line.split()
            assert(params[1] == 'cnf') #we should be reading an unweighted SAT problem
            nbvar = int(params[2]) #number of variables
            break            
        else:
            assert(line[0] == 'c')
    f.close()

    return nbvar
    
def nCr(n, r):
    '''
    Compute n choose r
    https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    '''
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

def bigFloat_nCr(n, r):
    '''
    Outputs:
    - ret_val: bigFloat, n choose r
    '''
    ret_val = bf.factorial(n)/(bf.factorial(r)*bf.factorial(n-r))
    return ret_val

def worstCaseSet_sumAxZero(n, m, k, f, DEBUG=False, only_even_w=False):
    """Sum over worst case set of probability of A(x - x') = 0, A of size @m x @n
    is first chosen to have, on each row, @k entries that are one such that each column has at most a single 1,
    then all entries are flippedmwith probability @f (i.e. random walk has length w).

    Output:
    - worstCase_setSize: list of ints, worst case set sizes (from smallest to largest)
    - sumAxZero: list of floats, same length as worstCase_setSize, sum over worst case set of probability of A(x - x') = 0
        for the corresponding set size in worstCase_setSize

    - sorted_probAxZero: list of floats, each element is p[A(x - x') = 0] for all vectors of some hamming weight.  The
        list is sorted from largest probability to smallest
    - set_sizes_sortedByProb: list of ints (same length as sorted_probAxZero), each element is the number of vectors
        that have hamming weight for the corresponding probability in sorted_probAxZero 

    """
    USE_BIGFLOAT = False
    set_sizes = []
    set_probAxZero = []
    w_vals = []
    binom = BigBinom(n)
    USE_TRIS_SUPER_FAST_IMPLEMENTATION = True
    if USE_TRIS_SUPER_FAST_IMPLEMENTATION:
        all_ln_prob = get_Ax_zero_log_probs_all(n, m, k, f)
        assert(len(all_ln_prob) == n)
        for idx, cur_ln_prob in enumerate(all_ln_prob):
            w = idx + 1
            w_vals.append(w)
            if only_even_w and w%2 == 1:
                cur_prob = 0.0
            else:   
                cur_prob = math.exp(cur_ln_prob)
    
            set_probAxZero.append(cur_prob)
            if USE_BIGFLOAT:
                set_sizes.append(bigFloat_nCr(n, w))
            else:
                set_sizes.append(nCr(n, w))

    else:
        for w in range(1, n+1):
            w_vals.append(w)
            if only_even_w and w%2 == 1:
                cur_prob = 0.0
            else:   
                #cur_prob = get_Ax_zero_probs(n, m, w, k, f)
                cur_ln_prob = get_Ax_zero_log_probs(n, m, w, k, f)
                cur_prob = math.exp(cur_ln_prob)
    
    #        print cur_prob
            set_probAxZero.append(cur_prob)
            if USE_BIGFLOAT:
                set_sizes.append(bigFloat_nCr(n, w))
            else:
                set_sizes.append(nCr(n, w))


    sorted_probAxZero, set_sizes_sortedByProb, sorted_w_vals = zip(*sorted(zip(set_probAxZero, set_sizes, w_vals), reverse=True))
    if DEBUG:
        print set_probAxZero
        print
        print 'sorted probs:', sorted_probAxZero
    
        print set_sizes
        print
        print 'sorted set sizes:', set_sizes_sortedByProb
    worstCase_setSize = [set_sizes_sortedByProb[0]]
    sumAxZero = [set_sizes_sortedByProb[0]*sorted_probAxZero[0]]
    assert(len(set_sizes) == len(set_probAxZero))
    for idx in range(1, len(set_sizes)):
        worstCase_setSize.append(worstCase_setSize[-1] + set_sizes_sortedByProb[idx])
        sumAxZero.append(sumAxZero[-1] + set_sizes_sortedByProb[idx]*sorted_probAxZero[idx])
    assert(worstCase_setSize[-1] == 2**n - 1)
    if DEBUG:
        print 'simple debug'
        print '#'*80
        print worstCase_setSize[-1] 
        print 2**n - 1
        print '#'*80
        print w_vals
        print sorted_w_vals

    return (sorted_probAxZero, set_sizes_sortedByProb, sorted_w_vals)
    #return (worstCase_setSize, sumAxZero)


class BigBinom:
    """ Class that computes binomial and partial sum of binomials and returns values in BigFloat
        This class once computed a result, caches it, after which each query only takes O(1) time """

    def __init__(self, n):
        self.n = n
        # binom_list[m] = binom(n, m)
        self.binom_list = []
        # partial_binom_sum[m] = \sum_{w = 0}^{m} binom(n, w)
        self.partial_binom_sum = [BigFloat(1)]

        # partial_binom_sum[m] = \sum_{even w = 0}^{m} binom(n, w)        
        self.partial_binom_sum_even = [BigFloat(1)]

    def binom(self, m):
        # If result not yet computed, compute and cache it
        if len(self.binom_list) < m + 1:
            for w in range(len(self.binom_list), m + 1):
                self.binom_list.append(BigBinom.factorial(self.n) / (BigBinom.factorial(w) * BigBinom.factorial(self.n - w)))
        return self.binom_list[m]

    def binom_sum(self, m):
        # If result not yet computed, compute and cache it
        if len(self.binom_list) < m + 1:
            for w in range(len(self.binom_list), m + 1):
                self.binom_list.append(BigBinom.factorial(self.n) / (BigBinom.factorial(w) * BigBinom.factorial(self.n - w)))
        if len(self.partial_binom_sum) < m + 1:
            for w in range(len(self.partial_binom_sum), m + 1):
                self.partial_binom_sum.append(self.partial_binom_sum[w - 1] + self.binom_list[w])
        return self.partial_binom_sum[m]

    def binom_sum_even(self, m):
        # sum of n choose w for even w from w=0 to m
        if len(self.binom_list) < m + 1:
            for w in range(len(self.binom_list), m + 1):
                self.binom_list.append(BigBinom.factorial(self.n) / (BigBinom.factorial(w) * BigBinom.factorial(self.n - w)))
        if len(self.partial_binom_sum_even) < m + 1:
            for w in range(len(self.partial_binom_sum_even), m + 1):
                if w%2 == 0:
                    self.partial_binom_sum_even.append(self.partial_binom_sum_even[w - 1] + self.binom_list[w])
                else:
                    self.partial_binom_sum_even.append(self.partial_binom_sum_even[w - 1])
        return self.partial_binom_sum_even[m]


    @staticmethod
    def factorial(n, approx_threshold=100):
        # start_time = time.time()
        if approx_threshold < 0 or n < approx_threshold:
            result = bf.factorial(n)
        else:
            result = math.sqrt(2 * math.pi * n) * bf.pow(n / math.e, n)
        # end_time = time.time()
        # print(end_time - start_time)
        return result

def prob_Ax0_for_hamming_weight(n, m, w, f):
    '''
    returns P[Ax=0] (i.e. r^(w,f)) when A is permuted block diag w/ deterministic k=1
    Inputs:
    - n: number of variables
    - m: number of parity constraints
    - w: hamming weight
    - f: probability of flipping 0s to 1's
    '''

    NUMERICAL_METHOD = 'BigFloat'
    #NUMERICAL_METHOD = 'Logs'
    #NUMERICAL_METHOD = 'Original'
    BF_PRECISION = 200
    if NUMERICAL_METHOD == 'BigFloat':
        with bf.precision(BF_PRECISION):
            cur_prob_Ax_zero = 0
            for collision_count in range(max(0, m-(n-w)), min(w, m)+1):
                cur_prob_Ax_zero += bigFloat_nCr(m, collision_count) * bigFloat_nCr(n - m, w - collision_count)\
                                    * ((.5 + .5*(1-2*f)**w)**(m-collision_count)) * ((.5 - .5*(1-2*f)**(w-1))**collision_count)\
                                    / bigFloat_nCr(n, w)
            return cur_prob_Ax_zero

    prob = 0
    total_vec_count = nCr(n, w)            
    for collision_count in range(max(0, w-(n-m)), min(w, m) + 1):
        cur_vec_count = nCr(m, collision_count)*nCr(n-m, w-collision_count)
        cur_prob = cur_vec_count/total_vec_count
        prob += cur_prob * ((.5 + .5*(1-2*f)**w)**(m-collision_count)) * ((.5 - .5*(1-2*f)**(w-1))**collision_count)

    print '#'*10
    print 'n =', n, 'm =', m, 'w =', w, 'f =', f, 
    print 'new p[Ax=0]:', prob, 'old p[Ax=0]:', (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** w)) ** m, 'difference =', (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** w)) ** m - prob 
    return prob

class upperBoundSATCount:
    def __init__(self, n, m, f, verbose=True, deterministicK1_permutation=False, only_even_w=False, k=0, USE_TRIS_CODE=True):
        """ Each SATCounter solves a specific sat problem that must be specified at creation 

        Inputs:
        - only_even_w: bool, calculate when worst case hamming ball only includes vectors with
            even hamming distance, possible when adding parity constraint of all variables

        """
        self.n = n
        self.verbose = verbose
        self.binom = BigBinom(n)
        self.deterministicK1_permutation = deterministicK1_permutation #where columns of A permuted?
        self.USE_TRIS_CODE = USE_TRIS_CODE #more general, can have any block size k permuted
        self.k = k #A has blocks of size k, only used with Tri's code        
        self.only_even_w = only_even_w
        self.f = f
        self.m = m

        if USE_TRIS_CODE:
            (self.sorted_probAxZero, self.set_sizes_sortedByProb, sorted_w_vals) = worstCaseSet_sumAxZero(n=self.n, m=self.m, k=self.k, f=self.f, only_even_w=self.only_even_w)

    def upper_bound_expected(self, m, tolerance=0.001):
        """ 
        Calcluate an upper bound on the set size that holds if half the trials find an empty bin,
        given m, and f, accurate to given tolerance.

        Bound holds with probability 1 - e^(-T/24), where T is the number of trials or SAT problems run
        """

        # Shorthand definition
        two_to_m = BigFloat(2.0) ** m

        # Use binary search to find the minimum q so that z > 3/4
        q_min = BigFloat(1.0)
        q_max = BigFloat(2.0) ** self.n
        for iteration in range(0, self.n + 10):
            q_mid = bf.sqrt(q_min * q_max)  # Search by geometric mean
            if self.USE_TRIS_CODE:
                v = q_mid / two_to_m * (1 + self.compute_eps_q_useTrisCode(m, q_mid, self.f) - q_mid / two_to_m)
            else:
                v = q_mid / two_to_m * (1 + self.compute_eps_q_mod(m, q_mid, self.f) - q_mid / two_to_m)
            z = 1 - v / (v + (q_mid / two_to_m) ** 2)
            if z > 3.0 / 4:
                q_max = q_mid
            else:
                q_min = q_mid

            # If difference between q_min and q_max is less than tolerance, stop the search
            if q_max < q_min * (1 + tolerance):
                break
        return bf.sqrt(q_max * q_min)



    def compute_eps_q_original(self, m, q, f):
        """ This function computes epsilon(n, m, q, f) * (q - 1), and is optimized for multiple queries """
        # Use binary search to find maximum w_star so that sum_{w = 1}^{w_star} C(n, w) <= q - 1
        # The possible w_star lies in [w_min, w_max]
        w_min = 0
        w_max = self.n
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

    def compute_eps_q_mod(self, m, q, f):
        """ This function computes epsilon(n, m, q, f) * (q - 1), and is optimized for multiple queries """
        # Use binary search to find maximum w_star so that sum_{w = 1}^{w_star} C(n, w) <= q - 1
        # The possible w_star lies in [w_min, w_max]
        w_min = 0
        w_max = self.n
        while w_min != w_max:
            w = int(math.ceil(float(w_min + w_max) / 2))    # If w_min + 1 = w_max, assign w to w_max
            if self.only_even_w:
                cur_sum = self.binom.binom_sum_even(w)
            else:
                cur_sum = self.binom.binom_sum(w)
            if cur_sum < q:
                w_min = w
            elif cur_sum == q:
                w_min = w
                break
            else:
                w_max = w - 1
        w_star = w_min


        if self.only_even_w:
            r = q - 1 - self.binom.binom_sum_even(w_star)
        else:
            r = q - 1 - self.binom.binom_sum(w_star)

        compute_qMinus2 = r

        if self.deterministicK1_permutation:
            epsq = r * prob_Ax0_for_hamming_weight(n=self.n, m=m, w=w_star+1, f=f)
            for w in range(1, w_star + 1):
                if (w%2 == 0) or not self.only_even_w:
                    epsq += self.binom.binom(w) * prob_Ax0_for_hamming_weight(n=self.n, m=m, w=w, f=f)
                    compute_qMinus2 += self.binom.binom(w)
        else:

            # Compute eps * (q - 1)
            epsq = r * (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** (w_star + 1))) ** m
            for w in range(1, w_star + 1):
                if (w%2 == 0) or not self.only_even_w:
                    epsq += self.binom.binom(w) * (0.5 + 0.5 * (BigFloat(1.0 - 2 * f) ** w)) ** m
                    compute_qMinus2 += self.binom.binom(w)

#        assert(compute_qMinus2 == q-2), (compute_qMinus2, q-2, w_star)
        return epsq


    def compute_eps_q_useTrisCode(self, m, q, f):
        """ This function computes epsilon(n, m, q, f) * (q - 1), and is optimized for multiple queries """

        #assert(self.only_even_w == False), "haven't implemented only_even_w yet!"
        #(worstCase_setSize, sumAxZero) = worstCaseSet_sumAxZero(n=self.n, m=m, k=SET_ME, f=f)
        epsq = 0
        cur_wcSet_size = 1 #current worst case set size
        for (wc_idx, wc_size) in enumerate(self.set_sizes_sortedByProb): #wc_size is the number of vectors with the worst case hamming weight indexed by wc_idx
            next_wcSet_size = cur_wcSet_size + wc_size #add number of vectors that have next worst hamming weight
            if next_wcSet_size < q: 
                epsq += wc_size * self.sorted_probAxZero[wc_idx]
                cur_wcSet_size = next_wcSet_size
            elif next_wcSet_size == q:
                epsq += wc_size * self.sorted_probAxZero[wc_idx]
                return epsq
            else:
                assert(next_wcSet_size > q)
                final_hammingWeight_vectorCount = q - cur_wcSet_size
                epsq += final_hammingWeight_vectorCount * self.sorted_probAxZero[wc_idx]
                return epsq


    @staticmethod
    def compute_eps_q_static_original(n, m, q, f, binom=None):
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



def get_parallel_upper_bound(num_UNSAT, T, n, m, f, k, UNSAT_runtimes, all_runtimes, params, min_confidence=.95):
    '''
    
    Inputs:
    - num_UNSAT: int, the number of trials we found to be UNSAT
    - T: int, the number of trials
    - n: int, the number of variables
    - m: int, the number of parity constraints
    - f: float, probability with which elements are flipped
    - k: int, block size of 1's that are first set to 1
    - min_confidence: float, the probability with which the bound holds
    - UNSAT_runtimes: list of floats of length num_SAT,
        where each entry corresponds to the runtime for solving that problem instance

    Outputs:
    - ln_upperBound: float, ln(upper bound)
    - parallel_runtime: float, parallel runtime required to solve SAT problems for this bound.
        We need to find that more than half the problems are UNSAT.  Sort the runtimes of all UNSAT
        problems and then find the time that allows just more than half of the problems to found as UNSAT.
    '''

    #Bound holds with probability 1 - e^(-T/24), where T is the number of trials (or SAT problems run)
    if (1 - np.exp(-T/24) < min_confidence):
        print 'cannot compute a bound with the required min_confidence, too few trials (only', T, ')'
        return (None, None)

    if num_UNSAT/T <= .5:
        return (None, None) #cannot compute an upper bound 

    num_UNSAT_needed = math.ceil(T/2)
    if num_UNSAT_needed == T/2:
        num_UNSAT_needed += 1
    num_UNSAT_needed = int(num_UNSAT_needed)
    UNSAT_runtimes.sort()
    parallel_runtime = UNSAT_runtimes[num_UNSAT_needed - 1]
    print 'runtime= %f, meanruntime= %f' % (parallel_runtime, np.mean(all_runtimes))

#    if k == 0:
#        satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=False, only_even_w=False, k=k, USE_TRIS_CODE=False)        
#        ln_upperBound = math.log(satProblem.upper_bound_expected(m=m))
#
#    elif k == 1 and params['f_block'] == 1:
    if k == 1 and params['f_block'] == 1:
        satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=True, only_even_w=False, k=k, USE_TRIS_CODE=False)        
        ln_upperBound = math.log(satProblem.upper_bound_expected(m=m))

    else:
        assert((k >= 1 and params['f_block'] == '1minusF') or (k == 0))
        satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=False, only_even_w=False, k=k, USE_TRIS_CODE=True)        
        ln_upperBound = math.log(satProblem.upper_bound_expected(m=m))

    
    return (ln_upperBound, parallel_runtime)

def get_all_parallel_upperBounds(params, repeats=REPEATS, file_count=FILE_COUNT):
    filebase = get_filebase(params)
    n = get_n(PROBLEM_NAME)
    print 'n=', n
    (sorted_m_vals, sorted_f_vals, SAT_runtimes, UNSAT_runtimes, num_SAT, num_trials_dict, all_runtimes) = read_files_moreInfo(filename_base=filebase, repeats=REPEATS, file_count=FILE_COUNT)
    parallelBounds = []
    parallelRuntimes = []
    #for (f_idx, f_val) in enumerate((sorted_f_vals)):
    for f_val in [i/100 for i in range(15, 51)]:
        for (m_idx, m_val) in enumerate(sorted_m_vals):    
            #if f_val > .15:
            #    break
            if params['k'] == None or params['k'] == 'maxConstant':
                k = int(np.floor(n/m_val))
            else:
                k = params['k']
            #assert(float(int(num_SAT[f_idx, m_idx])) == num_SAT[f_idx, m_idx])
            (ln_upperBound, parallel_runtime) = \
                get_parallel_upper_bound(num_UNSAT=len(UNSAT_runtimes[(f_val, m_val)]), T=num_trials_dict[(f_val, m_val)],\
                    n=n, m=m_val, f=f_val, k=k, UNSAT_runtimes=UNSAT_runtimes[(f_val, m_val)], all_runtimes=all_runtimes[(f_val, m_val)], params=params, min_confidence=.95)
            if ln_upperBound != None:
                print 'numUNSAT=%d, Trials=%d, m=%d, f=%f, upperbound=%f, runtime=%f' % (len(UNSAT_runtimes[(f_val, m_val)]), num_trials_dict[(f_val, m_val)], m_val, f_val, ln_upperBound, parallel_runtime)
                parallelBounds.append(ln_upperBound)
                parallelRuntimes.append(parallel_runtime)
    return parallelBounds, parallelRuntimes

def plot_UBvsRuntime():
    
    color_list = ['r', 'g', 'b']#, 'y', 'p']
    for color_idx, params in enumerate(all_params_to_plot):
        (parallel_bounds, parallel_runtimes) = \
            get_all_parallel_upperBounds(params=params)


        if PLOT_ACTUAL_POINTS:
            plt.scatter(parallel_runtimes, parallel_bounds, marker='+', c=color_list[color_idx])
            plt.xlabel('parallel runtime (units: mean unperturbed runtime on machine averaged over 10 trials)')
            plt.ylabel('bound on ln(set size)')
            plt.legend()
        #    plt.show()

        PLOT_PARALLEL_CONVEX_HULLS = True
        if PLOT_PARALLEL_CONVEX_HULLS:
            #plot convex hull of points from original method
            original_points = np.array(zip(parallel_runtimes,parallel_bounds))
            original_hull = ConvexHull(original_points)

            #print "original_points:"
            #print original_points

            for (idx, simplex) in enumerate(original_hull.simplices):
                #print 'idx:', idx, "simplex:", simplex, 'len(original_hull.simplices):', len(original_hull.simplices)
                #print "type(simplex):", type(simplex)
                #print original_points[simplex]
                #print original_points[simplex, 0]
                #print original_points[simplex, 1]
                if idx == 0: #add the label to the legend
                    print 'color:', color_list[color_idx]
                    plt.plot(original_points[simplex, 0], original_points[simplex, 1], '*--', c=color_list[color_idx], \
                        label='f_blk:%s, permute:%s, k:%s, all1constraint:%s' % \
                        (params['f_block'], params['permute'], params['k'], params['allOnesConstraint']))
                else:
                    plt.plot(original_points[simplex, 0], original_points[simplex, 1], '*--', c=color_list[color_idx])
                if ANNOTATE_PLOTS:
                    m_val0 = parallel_orig_M_F[simplex[0]][0]
                    f_val0 = parallel_orig_M_F[simplex[0]][1]

                    m_val1 = parallel_orig_M_F[simplex[1]][0]
                    f_val1 = parallel_orig_M_F[simplex[1]][1]
                    plt.annotate(
                        'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f' % (m_val0, f_val0, all_original_num_trials[(f_val0, m_val0)], \
                            orig_sat_over_trials_parallel[simplex[0]], orig_satUsed_over_totalSat_parallel[simplex[0]]),
                        xy=(original_points[simplex[0], 0], original_points[simplex[0], 1]), xytext=(-20, 20),
                        textcoords='offset points', ha='right', va='bottom', color=color_list[color_idx],
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color=color_list[color_idx]))
                    plt.annotate(
                        'm:%d,f:%.3f,T:%d,c:%.2f,%%c:%.2f'% (m_val1, f_val1, all_original_num_trials[(f_val1, m_val1)], \
                            orig_sat_over_trials_parallel[simplex[1]], orig_satUsed_over_totalSat_parallel[simplex[1]]),
                        xy=(original_points[simplex[1], 0], original_points[simplex[1], 1]), xytext=(-20, 20),
                        textcoords='offset points', ha='right', va='bottom', color=color_list[color_idx],
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', color=color_list[color_idx]))
     
    plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='ground truth ln(set size)') 
    plt.xlabel('parallel runtime (units: mean unperturbed runtime on machine averaged over 10 trials)')
    plt.ylabel('bound on ln(set size)')
    plt.legend()    
    plt.show()



if __name__=="__main__":
    plot_UBvsRuntime()
    exit(0)

    n = 1000
    f = .06
    m = 20
    for k in range(0, 51):
        f_prime = (n*f - k)/(n-2*k)
        print 'Tri code'
        #for (m) in [(150)]:
        start_time = time.time()        
        satProblem = upperBoundSATCount(n=n, m=m, f=f_prime, deterministicK1_permutation=False, only_even_w=False, k=k, USE_TRIS_CODE=True)        
        print 'n =', n, 'm =', m, 'f =', f_prime, 'k =', k, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
        end_time = time.time()        
        print 'time:', end_time - start_time

    exit(0)

    n = 300
    f = .05
    m=50
    k=6

    print 'orig, k=0'
    start_time = time.time()
    satProblem = upperBoundSATCount(n=n, m=m, f=f+k/n, deterministicK1_permutation=False, only_even_w=False, k=0, USE_TRIS_CODE=False)        
    print 'n =', n, 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
    end_time = time.time()        
    print 'time:', end_time - start_time
    print 

    print 'permutation, k=1'
    start_time = time.time()            
    satProblem = upperBoundSATCount(n=n, m=m, f=f+(k-1)/n, deterministicK1_permutation=True, only_even_w=False, k=0, USE_TRIS_CODE=False)
    print 'n =', n, 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
    end_time = time.time()        
    print 'time:', end_time - start_time
    print 


    print 'Tri code'
    #for (m) in [(150)]:
    start_time = time.time()        
    satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=False, only_even_w=False, k=k, USE_TRIS_CODE=True)        
    print 'n =', n, 'm =', m, 'f =', f, 'k =', k, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
    end_time = time.time()        
    print 'time:', end_time - start_time
    print 

    ####no need to save cach with Tri's new super fast implementation
    ###cache some more values
    ##with open('.cache', 'wb') as cache_file:
    ##    pickle.dump(log_g.cache, cache_file)

    exit(0)

    #f = .05

####    start_time = time.time()
####    cur_ln_prob = get_Ax_zero_log_probs(n=n, m=10, w=50, k=10, f=f)    
####    end_time = time.time()        
####    print 'first call time:', end_time - start_time
####
####    start_time = time.time()
####    cur_ln_prob = get_Ax_zero_log_probs(n=n, m=10, w=50, k=10, f=f)    
####    end_time = time.time()        
####    print 'second call time:', end_time - start_time
####
####    exit(0)
    #m = 5
    #k = 14
    #f = .01
    #worstCaseSet_sumAxZero(n, m, k, f, DEBUG=True, only_even_w=True)
    #exit(0)
    print 'orig, k=0'
    f=.13
    m=37
    start_time = time.time()
    satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=False, only_even_w=False, k=0, USE_TRIS_CODE=False)        
    print 'n =', n, 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
    end_time = time.time()        
    print 'time:', end_time - start_time
    print 
      
##########
##########    print 'orig, k=0'
##########    for (m) in [(10)]:
##########        start_time = time.time()    
##########        satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=False, only_even_w=False, k=0, USE_TRIS_CODE=False)
##########        print 'n =', n, 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
##########        end_time = time.time()        
##########        print 'time:', end_time - start_time
##########        print 
##########
##########    start_time = time.time()        
##########
##########    for (m) in [(10)]:
##########        print 'Tri code k=0'
##########        satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=False, only_even_w=False, k=0, USE_TRIS_CODE=True)
##########        print 'n =', n, 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
##########        end_time = time.time()        
##########        print 'time:', end_time - start_time
##########        print 
##########
#############    print 'permutation, k=1'
#############    satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=True, only_even_w=False, k=0, USE_TRIS_CODE=False)
#############    for (m) in [(10)]:
#############        start_time = time.time()        
#############        print 'n =', n, 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
#############        end_time = time.time()        
#############        print 'time:', end_time - start_time
#############        print 

    print 'Tri code'
    #for (m) in [(150)]:
    f=.13
    m=37
    k = 5
    start_time = time.time()        
    satProblem = upperBoundSATCount(n=n, m=m, f=f, deterministicK1_permutation=False, only_even_w=False, k=k, USE_TRIS_CODE=True)        
    print 'n =', n, 'm =', m, 'f =', f, 'k =', k, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m))/math.log(2)
    end_time = time.time()        
    print 'time:', end_time - start_time
    print 

#double check only_even_w!!

    exit(0)

    #lang12
    #satProblem = upperBoundSATCount(n=576)
    #for (m, f) in [(18, .05), (18, .04), (18, .03), (16, .02), (13, .01)]:

#####    #c432
#####    satProblem = upperBoundSATCount(n=196)
#####    for (m, f) in [(37, .07), (37, .06), (37, .05), (36, .04), (34, .03)]:
#####        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)
#####
#####    print '-'*80
#####
#####    #c432
#####    satProblem = upperBoundSATCount(n=196, deterministicK1_permutation=True)
#####    for (m, f) in [(37, .07), (37, .06), (36, .05), (36, .04), (35, .03), (33, .02)]:
#####        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)
#####


    print '-'*40, 'demo', '-'*40

    #sat-grid-pbl-0010
    print 'orig'
    satProblem = upperBoundSATCount(n=1000, deterministicK1_permutation=False)
    for (m, f) in [(10, .05)]:
        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)

    print 'orig'
    satProblem = upperBoundSATCount(n=1000, deterministicK1_permutation=False, only_even_w=True)
    for (m, f) in [(10, .051)]:
        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)


    print 'orig, even hamming ball'
    satProblem = upperBoundSATCount(n=1000, deterministicK1_permutation=False, only_even_w=True)
    for (m, f) in [(10, .05)]:
        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)


    print 'new'
    satProblem = upperBoundSATCount(n=1000, deterministicK1_permutation=True)
    for (m, f) in [(50, .0066)]:
        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)

    exit(0)



    print '-'*40, 'sat-grid-pbl-0010', '-'*40

    #sat-grid-pbl-0010
    print 'orig'
    satProblem = upperBoundSATCount(n=110, deterministicK1_permutation=False)
    for (m, f) in [(79, .06), (79, .05), (67, .04), (65, .03)]:
        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)

    print 'new'
    satProblem = upperBoundSATCount(n=110, deterministicK1_permutation=True)
    for (m, f) in [(81, .06), (80, .05), (79, .04), (76, .03), (65, .02)]:
        print 'm =', m, 'f =', f, 'upper bound =', math.log(satProblem.upper_bound_expected(m=m, f=f))/math.log(2)



