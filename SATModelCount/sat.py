from __future__ import division
__author__ = 'shengjia'

import subprocess, threading
import random
import time
import os
import math
import numpy as np
import copy
import heapq
import sklearn.feature_selection

try:
    import cPickle as pickle
except ImportError:
    import pickle

MACHINE = 'atlas'#'atlas' #'atlas' or 'local'

if MACHINE == 'local':
    CRYPTOMINISAT5_DIRECTORY = '/Users/jkuck/software/cryptominisat-5.0.1/build'
    SAT_SOLVER = "CRYPTOMINISAT5"
    #SAT_SOLVER = "ORIGINAL"
    SAT_PROBLEM_FOLDER = '../../winter_2018/low_density_parity_checks/SAT_problems_cnf'
else:
    import matplotlib
    matplotlib.use('Agg') #prevent error running remotely
    import matplotlib.pyplot as plt
    INSTALL_DIRECTORY = '/atlas/u/jkuck'
    SAT_PROBLEM_FOLDER = '/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/'


# Runs a system command without timeout
def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return [item.strip() for item in iter(p.stdout.readline, b'')]


# Class that execute a system command with timeout
class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.process_lock = threading.Lock()
        self.output = ""

    def run(self, timeout):
        """ Run a system command with specified timeout. Set timeout < 0 if timeout is not required """
        def target():
            self.process_lock.acquire()
            if self.process is None:
                self.process = subprocess.Popen(self.cmd,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT)
                self.process_lock.release()
                self.output, err = self.process.communicate()
            else:
                self.process_lock.release()

        thread = threading.Thread(target=target)
        thread.start()
        if timeout > 0:
            thread.join(timeout)
        else:
            thread.join()
        if thread.is_alive():
            self.process_lock.acquire()
            if self.process is not None:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = False
            self.process_lock.release()
            thread.join()
            return None
        return self.output


class SAT:
    def __init__(self, problem_file, verbose=True, instance_id=-1, duplicate=0):
        '''

        Inputs:
        - duplicate: int, create this many duplicate copies of all variables.  
            after duplication Z' = Z^(duplicate + 1)
        '''
        # This id must be unique to this SAT instance
        if instance_id == -1:
            self.id = time.time() + np.random.rand()
        else:
            self.id = instance_id + str(np.random.rand())

        self.clauses = []
        # Read in the problem instance
        ifstream = open(problem_file)
        while True: #throw away comment lines
            header = ifstream.readline().split()
            if header[0] != 'c':
                break

        self.n = int(header[2])

        while True:
            curline = ifstream.readline()
            if not curline:
                break
            if curline[0] == 'c': #comment line
                continue
            self.clauses.append(curline.strip())
            #####jdk
            #print self.clauses[-1]
            #print type(self.clauses[-1])
            #print self.clauses[-1].split()
            #####jdk            
        self.clauseCount = len(self.clauses)
        if self.clauseCount != int(header[3]):
            print("Warning: clause count mismatch, expecting " + str(header[3]) + ", received " + str(self.clauseCount))


        #perform duplication
        self.duplicate_factor = duplicate+1
        new_clauses = []
        for cur_clause in self.clauses:
            #print '-'*20
            for literal in cur_clause.split():
                assert(np.abs(int(literal)) <= self.n)
                #print int(literal), np.abs(int(literal)), np.sign(int(literal))
        for dup_idx in range(duplicate):
            for cur_clause in self.clauses:
                new_clause = []
                for literal in cur_clause.split():
                    var = np.abs(int(literal))
                    if var > 0:
                        sign = np.sign(int(literal))
                        new_var = sign*(var + (dup_idx+1)*self.n)
                        new_clause.append(new_var)
                    else:
                        new_clause.append(0)
                new_clauses.append(' '.join(str(var) for var in new_clause))


        assert(len(new_clauses) == duplicate*self.clauseCount)
        self.clauses.extend(new_clauses)
        self.clauseCount *= (duplicate+1)
        assert(len(self.clauses) == self.clauseCount)
        self.n_unique = self.n        
        self.n *= (duplicate+1)

        # Set this to non-zero value to limit the maximum length of xor constraints
        # If this length is exceeded, xor will be broken up into separate clauses
        self.max_xor = -1
        self.new_variables = 0

        if verbose:
            print("CNF with " + str(self.n) + " variables and " + str(self.clauseCount) + " clauses")

        self.verbose = verbose
        self.hash_functions = []
        self.fail_apriori = False

    def construct_smaller_SAT_problem(self, restricting_hypercube):
        '''
        Make a new SAT problem that contains the satistfying solutions to the original SAT
        problem that fall within some smaller hypercube (fewer dimensions than original)

        Inputs:
        - restricting_hypercube: (dictionary) specification of the hypercube.  Each entry 
            specifies a dimension to cut in half:
                -key: (int) 1 indexed dimension
                -value: (0 or 1) value of dimension in the hypercube

        Outputs:
        - new_SAT_problem: (type SAT) the new SAT problem
        '''
        new_SAT_problem = copy.deepcopy(self)
        new_SAT_problem.id = str(new_SAT_problem.id) + str(np.random.rand())

        #renumber variables, so that new variables are contiguous
        new_variable_names = {}
        variable_present_new_problem = []
        for original_var_name in range(1, self.n+1):
            if original_var_name in restricting_hypercube:
                variable_present_new_problem.append(0)
            else:
                variable_present_new_problem.append(1)
        for original_var_name in range(1, self.n+1):
            if variable_present_new_problem[original_var_name-1] == 1:
                new_variable_names[original_var_name] = sum(variable_present_new_problem[:original_var_name])


        adjusted_clauses = []
        #perform duplication
        for cur_clause in new_SAT_problem.clauses:
            #print '-'*20
            for literal in cur_clause.split():
                assert(np.abs(int(literal)) <= new_SAT_problem.n)
                #print int(literal), np.abs(int(literal)), np.sign(int(literal))
        for cur_clause in new_SAT_problem.clauses:
            new_clause = []
            clause_satisfied = False
            for literal in cur_clause.split():
                var = np.abs(int(literal))
                literal_sign = np.sign(int(literal))
                if var in restricting_hypercube:
                    if (literal_sign == 1 and restricting_hypercube[var] == 1) or\
                       (literal_sign == -1 and restricting_hypercube[var] == 0):
                        clause_satisfied = True
                        break
                else:
                    if var > 0:
                        sign = np.sign(int(literal))
                        new_var = sign*(new_variable_names[var])
                        new_clause.append(new_var)
                    else:
                        new_clause.append(0)                    

            if clause_satisfied:
                continue
            elif len(new_clause) == 1: #only contains 0 signifying end of clause, no variables, so empty clause
                #print 'SAT problem unsatisfiable in construct_smaller_SAT_problem'
                #print cur_clause
                #for literal in cur_clause.split():
                #    var = np.abs(int(literal))
                #    print var, '=', restricting_hypercube[var]
                #exit(-1)
                new_SAT_problem.fail_apriori = True
            else:
                adjusted_clauses.append(' '.join(str(var) for var in new_clause))


        new_SAT_problem.clauses = adjusted_clauses
        new_SAT_problem.n = new_SAT_problem.n - len(restricting_hypercube)

        outside_hypercube_SAT_problem = copy.deepcopy(self)
        outside_clause = []
        for var_name, var_val in restricting_hypercube.iteritems():
            if var_val == 1:
                new_var = -1*var_name
                outside_clause.append(new_var)
            else:
                assert(var_val == 0)
                new_var = var_name
                outside_clause.append(new_var)
        assert(len(outside_clause) == len(restricting_hypercube))

        outside_hypercube_SAT_problem.clauses.append(' '.join(str(var) for var in outside_clause) + ' 0')
        #print ' '.join(str(var) for var in outside_clause) + ' 0'
        #exit(-1)
        return new_SAT_problem, outside_hypercube_SAT_problem, restricting_hypercube


    def add_parity_constraints(self, m, f):
        """ Add m parity constraints, each atom is included into the xor with probability f """
        if self.duplicate_factor == 1:
            self.hash_functions = []
            self.new_variables = 0
            self.fail_apriori = False
    
            cur_index = self.n + 1
    
            #print "n = ", self.n
            #print "clauseCount = ", self.clauseCount
            for i in range(0, m):
                new_function = []
    
                for atom in range(1, self.n + 1):
                    if random.random() < f:
                        new_function.append(atom)
                if len(new_function) == 0:
                    if random.randint(0, 1) == 0:
                        continue
                    else:
                        self.fail_apriori = True
                        return
                if random.randint(0, 1) == 0:
                    new_function[0] = -new_function[0]
                if self.max_xor > 0:
                    while len(new_function) > self.max_xor:
                        temp = new_function[0 : self.max_xor - 1]
                        new_function = [cur_index] + new_function[self.max_xor - 1:]
                        temp.append(cur_index)
                        cur_index += 1
                        self.new_variables += 1
                        self.hash_functions.append(temp)
                self.hash_functions.append(new_function)
            if self.verbose:
                print("Generated %d parity constraints" % m)
                if self.max_xor > 0:
                    print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))
        else:
            self.hash_functions = []
            self.new_variables = 0
            self.fail_apriori = False
    
            cur_index = self.n + 1
    
            #print "n = ", self.n
            #print "clauseCount = ", self.clauseCount
            for dup_idx in range(self.duplicate_factor):
                for i in range(0, m):
                    new_function = []
        
                    for atom in range(dup_idx*self.n_unique + 1, (dup_idx+1)*self.n_unique + 1):
                        if random.random() < f:
                            new_function.append(atom)
                    if len(new_function) == 0:
                        if random.randint(0, 1) == 0:
                            continue
                        else:
                            self.fail_apriori = True
                            return
                    if random.randint(0, 1) == 0:
                        new_function[0] = -new_function[0]
                    if self.max_xor > 0:
                        while len(new_function) > self.max_xor:
                            temp = new_function[0 : self.max_xor - 1]
                            new_function = [cur_index] + new_function[self.max_xor - 1:]
                            temp.append(cur_index)
                            cur_index += 1
                            self.new_variables += 1
                            self.hash_functions.append(temp)
                    self.hash_functions.append(new_function)
                if self.verbose:
                    print("Generated %d parity constraints" % m)
                    if self.max_xor > 0:
                        print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))

            print "number of parity constraints added:", len(self.hash_functions)

    def add_parity_constraints_proportional_to_marginals(self, m, f, marginals, variable_probs=np.array([None])):
        """ 
        Add m parity constraints, with average density of f, including each
        variable proportional to the uncertainty that it will be in a satisfying
        solution.  I.e. include variable x proportional to .5 - abs(.5-p(x))
    
        Inputs:
        - m: (int) nummber of parity constraints
        - f: (float) density of variables in parity constraints
        - marginals: (list of floats) approximate marginals for each variable,
            i.e. the probability that the variable is 1 in a satisfying solution
        - variable_probs: (np.array) with default of None, calculate from marginals.
            if supplied, ignore marginals and use specified probabilities for including
            each variable as specified explicitly
        """
        def get_variable_probs(marginals, f, n):
            unnormalized_probs = []
            for marginal in marginals:
                unnormalized_probs.append(.5 - abs(.5 - marginal))
            unnormalized_probs = np.array(unnormalized_probs)
            normalized_probs = unnormalized_probs/sum(unnormalized_probs)
            normalized_probs = normalized_probs*f*n #f*n constraints in expectation
            assert(abs(sum(normalized_probs) - f*n) < .00001), (sum(normalized_probs), f*n)
            print 'sum_normalized_probs = ', sum(normalized_probs), 'max(normalized_probs)', max(normalized_probs)            
            return normalized_probs

        def get_variable_probs1(marginals, f, n):
            unnormalized_probs = []
            for marginal in marginals:
                if marginal == 0.0 or marginal == 1.0:
                    unnormalized_probs.append(0.0)
                else:
                    unnormalized_probs.append(1.0)
            unnormalized_probs = np.array(unnormalized_probs)
            normalized_probs = unnormalized_probs/sum(unnormalized_probs)
            normalized_probs = normalized_probs*f*n #f*n constraints in expectation
            assert(abs(sum(normalized_probs) - f*n) < .00001), (sum(normalized_probs), f*n)
            print 'sum_normalized_probs = ', sum(normalized_probs), 'max(normalized_probs)', max(normalized_probs)
            return normalized_probs            

        if variable_probs.any() == None:
            assert(len(marginals) == self.n)
            variable_probs = get_variable_probs(marginals=marginals, f=f, n=self.n)
        else:
            assert(abs(sum(variable_probs)-1.0) < .00001)
        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        cur_index = self.n + 1

        #print "n = ", self.n
        #print "clauseCount = ", self.clauseCount
        total_vars_in_parity_constraints = 0        
        for i in range(0, m):
            new_function = []

            for atom in range(1, self.n + 1):
                if random.random() < variable_probs[atom-1]:
                    new_function.append(atom)
                    total_vars_in_parity_constraints += 1                    
            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    print 'self.fail_apriori = True'
                    for prob in variable_probs:
                        print prob
                    exit(0)
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0 : self.max_xor - 1]
                    new_function = [cur_index] + new_function[self.max_xor - 1:]
                    temp.append(cur_index)
                    cur_index += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)
        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))

        print 'empirical density = ', total_vars_in_parity_constraints/(self.n*m)






    def add_parity_constraints_restrict_vars(self, m, f, var_restriction):
        """
        Test whether variable overlap between parity constraints makes solving time faster or slower
        Add m parity constraints, each parity constraint contains f*n variables in expectation, restrict
        to including only a subset of variables in the parity constraints
        Inputs:
        - m: (int) the number of parity constraints
        - f: (float) each parity constraint contains f*n variables in expectation (n is total number of variables)
        - var_restriction: (int) only the first var_restriction variables can be included in any parity constraints
        """
        if var_restriction/self.n < f:
            print 'var_restriction is too small, cannot increase f sufficiently'
            return
        else: #compute new f so that we maintain the density f or variables in parity constraints, but restricted to a subset of variables
            new_f = self.n*f/var_restriction
            f = new_f

        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        cur_index = self.n + 1

        #print '-'*80        
        #print "n = ", self.n
        #print "clauseCount = ", self.clauseCount
        total_vars_in_parity_constraints = 0
        for i in range(0, m):
            new_function = []

            for atom in range(1, var_restriction + 1):
                if random.random() < f:
                    new_function.append(atom)
                    total_vars_in_parity_constraints += 1
            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0 : self.max_xor - 1]
                    new_function = [cur_index] + new_function[self.max_xor - 1:]
                    temp.append(cur_index)
                    cur_index += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)

        print 'empirical density = ', total_vars_in_parity_constraints/(self.n*m)
        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))


    def add_regular_constraints(self, m, f):
        """ Add m parity constraints, according to the new combined ensemble """
        if m * f <= 1:
            return self.add_parity_constraints(m, f)
        print("Using regular")


        k_low = int(math.floor(float(self.n) / m))
        k_high = int(math.ceil(float(self.n) / m))
        k_range = [0]
        for i in range(m):
            if (self.n - k_range[i]) % k_low == 0:
                k_high = k_low
            k_range.append(k_range[i] + k_high)
            
        f_updated = f - 1.0 / float(m)

        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        curIndex = self.n + 1

        for i in range(0, m):
            new_function = []

            for atom in range(1, self.n + 1):
                if random.random() < f_updated or (atom > k_range[i] and atom <= k_range[i+1]):
                    new_function.append(atom)
            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0:self.max_xor - 1]
                    new_function = [curIndex] + new_function[self.max_xor - 1:]
                    temp.append(curIndex)
                    curIndex += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)
        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))
   

    def add_regular_constraints_constantF(self, m, f):
        """ Add m parity constraints, according to the new combined ensemble without decreasing f """
        print("Using regular")


        k_low = int(math.floor(float(self.n) / m))
        k_high = int(math.ceil(float(self.n) / m))
        k_range = [0]
        for i in range(m):
            if (self.n - k_range[i]) % k_low == 0:
                k_high = k_low
            k_range.append(k_range[i] + k_high)
            
        f_updated = f 

        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        curIndex = self.n + 1

        for i in range(0, m):
            new_function = []

            for atom in range(1, self.n + 1):
                if random.random() < f_updated or (atom > k_range[i] and atom <= k_range[i+1]):
                    new_function.append(atom)
            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0:self.max_xor - 1]
                    new_function = [curIndex] + new_function[self.max_xor - 1:]
                    temp.append(curIndex)
                    curIndex += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)
        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))

    def add_regular_constraints_constantF_permuted(self, m, f, f_block, permute=True, k=None, \
        ADD_CONSTRAINT_ALL_ONES=False, change_var_names=True, variable_subset=None):
        """ 
        Add m parity constraints, according to the new combined ensemble without decreasing f,
        adding block 1's with probability f_block, 
        and permuting columns
        """
        if ADD_CONSTRAINT_ALL_ONES:
            m_effective = m - 1
        else:
            m_effective = m

        if k==None or k*m_effective > self.n: #use k = n/m_effective
            k_low = int(math.floor(float(self.n) / m_effective))
            k_high = int(math.ceil(float(self.n) / m_effective))
        else:
            k_low = k
            k_high = k
            
        number_k_high_blocks = self.n%m_effective
        k_range = [0]
        for i in range(number_k_high_blocks):
            k_range.append(k_range[i] + k_high)
        for i in range(number_k_high_blocks, m_effective):
            k_range.append(k_range[i] + k_low)            
        if k==None or k*m_effective > self.n: #use k = n/m_effective
            assert(k_range[-1] == self.n)
#        print k_range
        block_diag_matrix = np.zeros((m, self.n))
        #construct block diagonal 1's matrix
        for i in range(0, m):
            for atom in range(1, self.n+1):
                if ADD_CONSTRAINT_ALL_ONES:
                    if i == 0:
                        block_diag_matrix[i, atom-1] = 1
                    elif (atom > k_range[(i-1)] and atom <= k_range[(i-1)+1]):
                        block_diag_matrix[i, atom-1] = 1
                else:
                    if (atom > k_range[i] and atom <= k_range[i+1]):
                        block_diag_matrix[i, atom-1] = 1
#        print block_diag_matrix
        #permute the columns of the block diagonal matrix
        if permute and (not change_var_names):
            permuted_block_diag_matrix = np.swapaxes(np.random.permutation(np.swapaxes(block_diag_matrix,0,1)),0,1)
        elif permute and change_var_names:
            #permute columns of the parity constraint matrix, but keep blocks of ones by renaming the orginal variables
            permuted_vars_matrix = np.swapaxes(np.random.permutation(np.swapaxes(block_diag_matrix,0,1)),0,1)
            new_var_name = 1
            orig_var_names = set()
            #dictionary with key: original variable name (1 indexd), value: new variable name (1 indexd)
            new_var_names = {}
            for row in range(permuted_vars_matrix.shape[0]):
                for col in range(permuted_vars_matrix.shape[1]):
                    if permuted_vars_matrix[row, col] == 1:
                        new_var_names[col+1] = new_var_name
                        orig_var_names.add(col+1)
                        new_var_name+=1

            for orig_var_name in range(1, self.n+1):
                if not orig_var_name in orig_var_names:
                    new_var_names[orig_var_name] = new_var_name
                    new_var_name+=1
                    orig_var_names.add(orig_var_name)  
                           
            assert(len(orig_var_names) == self.n), (len(orig_var_names), self.n)
            assert(new_var_name == self.n + 1), (new_var_name, self.n + 1)

            renamed_clauses = []
            #perform duplication
            for cur_clause in self.clauses:
                new_clause = []
                for literal in cur_clause.split():
                    var = np.abs(int(literal))
                    literal_sign = np.sign(int(literal))
                    if var > 0:
                        if var in new_var_names:
                            new_literal = literal_sign*(new_var_names[var])
                        else:
                            new_literal = literal
                        new_clause.append(new_literal)
                    else:
                        new_clause.append(0)                    
                renamed_clauses.append(' '.join(str(var) for var in new_clause))
            self.clauses = renamed_clauses
            permuted_block_diag_matrix = block_diag_matrix

            #print new_var_names
            #print permuted_block_diag_matrix
            #prin
        else:
            permuted_block_diag_matrix = block_diag_matrix
#        print permuted_block_diag_matrix
        f_updated = f 

        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        curIndex = self.n + 1

        total_vars_in_parity_constraints = 0

        for i in range(0, m):
            new_function = []

            for atom in range(1, self.n + 1):
                #check block diagonal construction
                if ADD_CONSTRAINT_ALL_ONES:
                    if i == 0:
                        assert (block_diag_matrix[i, atom-1] == 1)
                    elif (atom > k_range[(i-1)] and atom <= k_range[(i-1)+1]):
                        assert (block_diag_matrix[i, atom-1] == 1)
                    else:
                        assert (block_diag_matrix[i, atom-1] == 0)
                else:
                    if (atom > k_range[i] and atom <= k_range[i+1]):
                        assert (block_diag_matrix[i, atom-1] == 1)
                    else:
                        assert (block_diag_matrix[i, atom-1] == 0)

                #is this element part of a permuted block?
                if (permuted_block_diag_matrix[i, atom-1] == 1):
                    #if so, add variable with probabilty f_block
                    if (random.random() < f_block and variable_subset == None) or\
                       (random.random() < f_block and atom in variable_subset):
                        new_function.append(atom)
                        total_vars_in_parity_constraints += 1
                #if this element isn't part of a permuted block, add variable with probability f_updated
                elif (random.random() < f_updated and variable_subset == None) or\
                  (random.random() < f_updated and atom in variable_subset):
                    new_function.append(atom)
                    total_vars_in_parity_constraints += 1

            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0:self.max_xor - 1]
                    new_function = [curIndex] + new_function[self.max_xor - 1:]
                    temp.append(curIndex)
                    curIndex += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)

        print 'empirical density = ', total_vars_in_parity_constraints/(self.n*m)

        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))


    def add_regular_constraints_constantF_permuted_debug(self, m, f, f_block, permute=True, k=None, \
        ADD_CONSTRAINT_ALL_ONES=False, change_var_names=True, variable_subset=None):
        """ 
        Add m parity constraints, according to the new combined ensemble without decreasing f,
        adding block 1's with probability f_block, 
        and permuting columns
        """
        if ADD_CONSTRAINT_ALL_ONES:
            m_effective = m - 1
        else:
            m_effective = m

        if k==None or k*m_effective > self.n: #use k = n/m_effective
            k_low = int(math.floor(float(self.n) / m_effective))
            k_high = int(math.ceil(float(self.n) / m_effective))
        else:
            k_low = k
            k_high = k
            
        number_k_high_blocks = self.n%m_effective
        k_range = [0]
        for i in range(number_k_high_blocks):
            k_range.append(k_range[i] + k_high)
        for i in range(number_k_high_blocks, m_effective):
            k_range.append(k_range[i] + k_low)            
        if k==None or k*m_effective > self.n: #use k = n/m_effective
            assert(k_range[-1] == self.n)
#        print k_range
        block_diag_matrix = np.zeros((m, self.n))
        #construct block diagonal 1's matrix
        for i in range(0, m):
            for atom in range(1, self.n+1):
                if ADD_CONSTRAINT_ALL_ONES:
                    if i == 0:
                        block_diag_matrix[i, atom-1] = 1
                    elif (atom > k_range[(i-1)] and atom <= k_range[(i-1)+1]):
                        block_diag_matrix[i, atom-1] = 1
                else:
                    if (atom > k_range[i] and atom <= k_range[i+1]):
                        block_diag_matrix[i, atom-1] = 1
#        print block_diag_matrix
        #permute the columns of the block diagonal matrix
        if permute and (not change_var_names):
            permuted_block_diag_matrix = np.swapaxes(np.random.permutation(np.swapaxes(block_diag_matrix,0,1)),0,1)
        elif permute and change_var_names:
            #permute columns of the parity constraint matrix, but keep blocks of ones by renaming the orginal variables
            permuted_vars_matrix = np.swapaxes(np.random.permutation(np.swapaxes(block_diag_matrix,0,1)),0,1)
            new_var_name = 1
            orig_var_names = set()
            #dictionary with key: original variable name (1 indexd), value: new variable name (1 indexd)
            new_var_names = {}
            for row in range(permuted_vars_matrix.shape[0]):
                for col in range(permuted_vars_matrix.shape[1]):
                    if permuted_vars_matrix[row, col] == 1:
                        new_var_names[col+1] = new_var_name
                        orig_var_names.add(col+1)
                        new_var_name+=1

            for orig_var_name in range(1, self.n+1):
                if not orig_var_name in orig_var_names:
                    new_var_names[orig_var_name] = new_var_name
                    new_var_name+=1
                    orig_var_names.add(orig_var_name)  
                           
            assert(len(orig_var_names) == self.n), (len(orig_var_names), self.n)
            assert(new_var_name == self.n + 1), (new_var_name, self.n + 1)

            renamed_clauses = []
            #perform duplication
            for cur_clause in self.clauses:
                new_clause = []
                for literal in cur_clause.split():
                    var = np.abs(int(literal))
                    literal_sign = np.sign(int(literal))
                    if var > 0:
                        if var in new_var_names:
                            new_literal = literal_sign*(new_var_names[var])
                        else:
                            new_literal = literal
                        new_clause.append(new_literal)
                    else:
                        new_clause.append(0)                    
                renamed_clauses.append(' '.join(str(var) for var in new_clause))
            self.clauses = renamed_clauses
            permuted_block_diag_matrix = block_diag_matrix

            #print new_var_names
            #print permuted_block_diag_matrix
            #prin
        else:
            permuted_block_diag_matrix = block_diag_matrix
#        print permuted_block_diag_matrix
        f_updated = f 

        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        curIndex = self.n + 1

        total_vars_in_parity_constraints = 0

        SAT_problem_debug = copy.deepcopy(self)

        for i in range(0, m):
            new_function = []
            new_function_debug = []
            for atom in range(1, self.n + 1):
                #check block diagonal construction
                if ADD_CONSTRAINT_ALL_ONES:
                    if i == 0:
                        assert (block_diag_matrix[i, atom-1] == 1)
                    elif (atom > k_range[(i-1)] and atom <= k_range[(i-1)+1]):
                        assert (block_diag_matrix[i, atom-1] == 1)
                    else:
                        assert (block_diag_matrix[i, atom-1] == 0)
                else:
                    if (atom > k_range[i] and atom <= k_range[i+1]):
                        assert (block_diag_matrix[i, atom-1] == 1)
                    else:
                        assert (block_diag_matrix[i, atom-1] == 0)

                #is this element part of a permuted block?
                if (permuted_block_diag_matrix[i, atom-1] == 1):
                    #if so, add variable with probabilty f_block
                    if random.random() < f_block:
                        new_function.append(atom)
                        total_vars_in_parity_constraints += 1
                        if (variable_subset != None) and (atom in variable_subset):
                            new_function_debug.append(atom)

                #if this element isn't part of a permuted block, add variable with probability f_updated
                elif random.random() < f_updated:
                    new_function.append(atom)
                    total_vars_in_parity_constraints += 1
                    if (variable_subset != None) and (atom in variable_subset):
                        new_function_debug.append(atom)
                        

            if len(new_function) > 0:                    
                if random.randint(0, 1) == 0:
                    new_function[0] = -new_function[0]
                if self.max_xor > 0:
                    while len(new_function) > self.max_xor:
                        temp = new_function[0:self.max_xor - 1]
                        new_function = [curIndex] + new_function[self.max_xor - 1:]
                        temp.append(curIndex)
                        curIndex += 1
                        self.new_variables += 1
                        self.hash_functions.append(temp)
                self.hash_functions.append(new_function)
            else:
                if random.randint(0, 1) == 1:
                    self.fail_apriori = True

            if len(new_function_debug) > 0:                    
                if random.randint(0, 1) == 0:
                    new_function_debug[0] = -new_function_debug[0]
                if SAT_problem_debug.max_xor > 0:
                    while len(new_function_debug) > SAT_problem_debug.max_xor:
                        temp = new_function_debug[0:SAT_problem_debug.max_xor - 1]
                        new_function_debug = [curIndex] + new_function_debug[SAT_problem_debug.max_xor - 1:]
                        temp.append(curIndex)
                        curIndex += 1
                        SAT_problem_debug.new_variables += 1
                        SAT_problem_debug.hash_functions.append(temp)
                SAT_problem_debug.hash_functions.append(new_function_debug)
            else:
                if random.randint(0, 1) == 1:
                    SAT_problem_debug.fail_apriori = True

        print 'empirical density = ', total_vars_in_parity_constraints/(self.n*m)

        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))

        return SAT_problem_debug

    def add_regular_constraints_subsetOfVariables(self, m, f, f_block, permute=True, k=None, \
        ADD_CONSTRAINT_ALL_ONES=False, change_var_names=True, variable_subset=[]):
        """ 
        Add m parity constraints, according to the new combined ensemble without decreasing f,
        adding block 1's with probability f_block, 
        and permuting columns
        """
        if variable_subset == []:
            variable_subset=[i+1 for i in range(self.n)]

        if ADD_CONSTRAINT_ALL_ONES:
            m_effective = m - 1
        else:
            m_effective = m

        if k==None or k*m_effective > len(variable_subset): #use k = n/m_effective
            k_low = int(math.floor(float(len(variable_subset)) / m_effective))
            k_high = int(math.ceil(float(len(variable_subset)) / m_effective))
        else:
            k_low = k
            k_high = k
            
        number_k_high_blocks = len(variable_subset)%m_effective
        k_range = [0]
        for i in range(number_k_high_blocks):
            k_range.append(k_range[i] + k_high)
        for i in range(number_k_high_blocks, m_effective):
            k_range.append(k_range[i] + k_low)            
        if k==None or k*m_effective > len(variable_subset): #use k = n/m_effective
            assert(k_range[-1] == len(variable_subset))
#        print k_range
        block_diag_matrix = np.zeros((m, len(variable_subset)))
        #construct block diagonal 1's matrix
        for i in range(0, m):
            for atom in range(1, len(variable_subset)+1):
                if ADD_CONSTRAINT_ALL_ONES:
                    if i == 0:
                        block_diag_matrix[i, atom-1] = 1
                    elif (atom > k_range[(i-1)] and atom <= k_range[(i-1)+1]):
                        block_diag_matrix[i, atom-1] = 1
                else:
                    if (atom > k_range[i] and atom <= k_range[i+1]):
                        block_diag_matrix[i, atom-1] = 1
#        print block_diag_matrix
        #permute the columns of the block diagonal matrix
        if permute and (not change_var_names):
            permuted_block_diag_matrix = np.swapaxes(np.random.permutation(np.swapaxes(block_diag_matrix,0,1)),0,1)
        elif permute and change_var_names:
            #permute columns of the parity constraint matrix, but keep blocks of ones by renaming the orginal variables
            permuted_vars_matrix = np.swapaxes(np.random.permutation(np.swapaxes(block_diag_matrix,0,1)),0,1)
            new_var_name = 1
            orig_var_names = set()
            #dictionary with key: original variable name (1 indexd), value: new variable name (1 indexd)
            new_var_names = {}
            for row in range(permuted_vars_matrix.shape[0]):
                for col in range(permuted_vars_matrix.shape[1]):
                    if permuted_vars_matrix[row, col] == 1:
                        new_var_names[col+1] = new_var_name
                        orig_var_names.add(col+1)
                        new_var_name+=1

            for orig_var_name in range(1, len(variable_subset)+1):
                if not orig_var_name in orig_var_names:
                    new_var_names[orig_var_name] = new_var_name
                    new_var_name+=1
                    orig_var_names.add(orig_var_name)  
                           
            assert(len(orig_var_names) == len(variable_subset)), (len(orig_var_names), len(variable_subset))
            assert(new_var_name == len(variable_subset) + 1), (new_var_name, len(variable_subset) + 1)

            renamed_clauses = []
            #perform duplication
            for cur_clause in self.clauses:
                new_clause = []
                for literal in cur_clause.split():
                    var = np.abs(int(literal))
                    literal_sign = np.sign(int(literal))
                    if var > 0:
                        if var in new_var_names:
                            new_literal = literal_sign*(new_var_names[var])
                        else:
                            new_literal = literal
                        new_clause.append(new_literal)
                    else:
                        new_clause.append(0)                    
                renamed_clauses.append(' '.join(str(var) for var in new_clause))
            self.clauses = renamed_clauses
            permuted_block_diag_matrix = block_diag_matrix

            #print new_var_names
            #print permuted_block_diag_matrix
            #prin
        else:
            permuted_block_diag_matrix = block_diag_matrix
#        print permuted_block_diag_matrix
        f_updated = f 

        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        curIndex = len(variable_subset) + 1

        total_vars_in_parity_constraints = 0

        for i in range(0, m):
            new_function = []

            for (var_idx, var_name) in enumerate(variable_subset):
                #check block diagonal construction
                if ADD_CONSTRAINT_ALL_ONES:
                    if i == 0:
                        assert (block_diag_matrix[i, var_idx] == 1)
                    elif (var_idx+1 > k_range[(i-1)] and var_idx+1 <= k_range[(i-1)+1]):
                        assert (block_diag_matrix[i, var_idx] == 1)
                    else:
                        assert (block_diag_matrix[i, var_idx] == 0)
                else:
                    if (var_idx+1 > k_range[i] and var_idx+1 <= k_range[i+1]):
                        assert (block_diag_matrix[i, var_idx] == 1)
                    else:
                        assert (block_diag_matrix[i, var_idx] == 0)

                #is this element part of a permuted block?
                if (permuted_block_diag_matrix[i, var_idx] == 1):
                    #if so, add variable with probabilty f_block
                    if random.random() < f_block:
                        new_function.append(var_name)
                        total_vars_in_parity_constraints += 1
                #if this element isn't part of a permuted block, add variable with probability f_updated
                elif random.random() < f_updated:
                    new_function.append(var_name)
                    total_vars_in_parity_constraints += 1

            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0:self.max_xor - 1]
                    new_function = [curIndex] + new_function[self.max_xor - 1:]
                    temp.append(curIndex)
                    curIndex += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)

        print 'empirical density = ', total_vars_in_parity_constraints/(len(variable_subset)*m)

        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))



    def add_permutation_constraints(self, m, f):
        """ Add m parity constraints, begin with rectangular permutation matrix with a single 1 per row and 0 or 1 1's per column,
         then add 1's with probability f """
        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        cur_index = self.n + 1

        #define the rectangular permutation matrix
        initial_ones = np.random.choice(range(1, self.n + 1), m, replace=False)

        for i in range(0, m):
            new_function = []

            for atom in range(1, self.n + 1):
                if initial_ones[i] == atom:
                    new_function.append(atom)
                elif random.random() < f:
                    new_function.append(atom)
            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0 : self.max_xor - 1]
                    new_function = [cur_index] + new_function[self.max_xor - 1:]
                    temp.append(cur_index)
                    cur_index += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)
        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))

    def add_variable_length_constraints(self, m, f):
        """ Test unevenness in length of constraints """
        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        cur_index = self.n + 1

        for i in range(0, m):
            new_function = []

            for atom in range(1, self.n + 1):
                if m == 0:
                    new_function.append(atom)                    
                elif random.random() < f:
                    new_function.append(atom)
            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.fail_apriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0 : self.max_xor - 1]
                    new_function = [cur_index] + new_function[self.max_xor - 1:]
                    temp.append(cur_index)
                    cur_index += 1
                    self.new_variables += 1
                    self.hash_functions.append(temp)
            self.hash_functions.append(new_function)
        if self.verbose:
            print("Generated %d parity constraints" % m)
            if self.max_xor > 0:
                print("Maximum xor length is %d. Added %d new variables" % (self.max_xor, self.new_variables))


    def add_halfspace_constraints(self, m, dimensions_to_cut):
        """ Add m constraints that cut space in half
        Inputs:
        - m: (int) number of cuts
        - dimensions_to_cut: (list of ints) each element is a 1 indexed dimension to cut
        """
        assert(len(dimensions_to_cut) == m)
        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        cur_index = self.n + 1

        #print "n = ", self.n
        #print "clauseCount = ", self.clauseCount
        for dimension in dimensions_to_cut:
            new_function = [dimension]
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            self.hash_functions.append(new_function)
        if self.verbose:
            print("Generated %d parity constraints" % m)


    def solve(self, max_time=-1, return_satisfying_solution=False):
        #print 'solve called'
        """ Attempt to solve the problem, returns True/False if the problem is satisfiable/unsatisfiable,
    returns None if timeout """
        if self.fail_apriori:
            if self.verbose:
                print("Constraint with zero length generated and inconsistent, UNSAT")
            if return_satisfying_solution:
                return ((False, 0), -1, None)
            else:
                return ((False, 0), -1)


        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        filename = "tmp/SAT_" + str(self.id) + ".cnf"
        ofstream = open(filename, "w")
        ofstream.write("p cnf " + str(self.n + self.new_variables) + " " + str(len(self.clauses) + len(self.hash_functions)) + "\n")

        for item in self.clauses:
            ofstream.write(item + "\n")
        parity_constraint_count = 0
        total_vars_in_parity_constraints = 0
        for hashFunction in self.hash_functions:
            parity_constraint_count += 1
            ofstream.write("x")
            for item in hashFunction:
                total_vars_in_parity_constraints += 1
                ofstream.write(str(item) + " ")
            ofstream.write("0\n")
        ofstream.close()

        #DEBUG
        #print os.getcwd()
        print 'wrote file:', filename
        print "hash_functions:", self.hash_functions
        #exit(0)
        #END DEBUG


        if parity_constraint_count > 0:
            empirical_density = total_vars_in_parity_constraints/(self.n*parity_constraint_count)
        else:
            empirical_density = 0
        start_time = time.time()
        if MACHINE == 'local':
            if SAT_SOLVER == 'ORIGINAL':
                solver = Command(['./cryptominisat', '--verbosity=0', '--gaussuntil=400', '--threads=1', filename])
            elif SAT_SOLVER == 'CRYPTOMINISAT5':
                RECONFIG = False
                if RECONFIG:
                    solver = Command(['%s/cryptominisat5' % CRYPTOMINISAT5_DIRECTORY, '--verb', '0', '--reconfat', '0', '--reconf', '7', filename])
                else:                    
                    solver = Command(['%s/cryptominisat5' % CRYPTOMINISAT5_DIRECTORY, '--verb', '0', filename])
        else:
            solver = Command(['%s/XORModelCount/SATModelCount/cryptominisat'%INSTALL_DIRECTORY, '--verbosity=0', '--gaussuntil=400', '--threads=1', filename])
        result = solver.run(timeout=max_time)
        end_time = time.time()
        run_command(['rm', filename])

        if self.verbose:
            if result is None:
                print("SAT solver timed out after " + str(max_time) + "s")
            else:
                print("SAT solved, time usage " + str(end_time - start_time) + "s, output: ")
                print("-----------------------------------------")
                print(result)
                print(type(result))
                print(len(result))
                print("-----------------------------------------")

        if result is None:
            return (None, empirical_density)

        result = result.split()
        if return_satisfying_solution and result[1] == 'SATISFIABLE':
            def RepresentsInt(s):
                try: 
                    int(s)
                    return True
                except ValueError:
                    return False     
            satisfying_solution = []    
            for possible_literal in result:
                if RepresentsInt(possible_literal):
                    literal = int(possible_literal)
                    if literal>0:
                        satisfying_solution.append(1)
                    elif literal<0:
                        satisfying_solution.append(0)
            assert(len(satisfying_solution) == self.n), (result, satisfying_solution, len(satisfying_solution), self.n)

        if len(result) >= 2:
            outcome = result[1]
            if outcome == 'SATISFIABLE':
                if return_satisfying_solution:
                    return ((True, end_time - start_time), empirical_density, satisfying_solution)                    
                else:
                    return ((True, end_time - start_time), empirical_density)
            elif outcome == 'UNSATISFIABLE':
                if return_satisfying_solution:
                    return ((False, end_time - start_time), empirical_density, None)
                else:
                    return ((False, end_time - start_time), empirical_density)

        print("Error: unrecognized return value")
        print("Full output: ")
        print(result)
        return None

def approximate_marginals_func(problem_name, repeats, f, m, verbose=True, REGULAR=False, PERMUTE=True):
    sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)    
    approximate_marginals = np.zeros(sat.n)
    total_satisfied_sol_found = 0

    runtimes = []
    SAT_runtimes = []
    for i in range(repeats):
        sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)

        if REGULAR:
            k = np.floor(sat.n/m)
            k_density = k/sat.n
            if k_density > f: #we need to decrease k
                k = np.floor(f*sat.n)

            f_prime = (f*sat.n - k)/(sat.n - 2*k)
            print 'f_prime:', f_prime
            sat.add_regular_constraints_constantF_permuted(m=m, f=f_prime, f_block=1-f_prime, permute=PERMUTE, k=k, ADD_CONSTRAINT_ALL_ONES=False)
      
        else:
            sat.add_parity_constraints(m=m, f=f)

        (outcome, empirical_density, satisfying_solution) = sat.solve(3600, return_satisfying_solution=True)
        if satisfying_solution:
            approximate_marginals += np.array(satisfying_solution)

        runtime = outcome[1]
        runtimes.append(runtime)

        if outcome[0] == True:
            total_satisfied_sol_found += 1
            SAT_runtimes.append(runtime)

    approximate_marginals/=total_satisfied_sol_found
    sequential_runtime = sum(runtimes)
    parallel_runtime = max(runtimes)
    if verbose:
        print 'approximate marginals with m=', m, 'f=', f, 'repeats=', repeats
        print 'sequential_runtime=', sequential_runtime, 'max runtime=', parallel_runtime, 'max SAT runtime=', max(SAT_runtimes)
        print approximate_marginals
    return approximate_marginals, sequential_runtime, parallel_runtime

    #approximate_marginals/=total_satisfied_sol_found
    #print 'approximate_marginals:'
    #print approximate_marginals
    #unique, counts = np.unique(approximate_marginals, return_counts=True)
    #count_dict = dict(zip(unique, counts))
    #marginalCount_0_1 = 0
    #if 0.0 in count_dict:
    #    marginalCount_0_1 += count_dict[0.0]
    #if 1.0 in count_dict:
    #    marginalCount_0_1 += count_dict[1.0]
    #print 'number of 0 or 1 approximate_marginals =', marginalCount_0_1
    #exit(0)

def solve_smaller_sat(problem_name, repeats, f, m):
    sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)    

    approximate_marginals, approximate_marginals_sequential_runtime, approximate_marginals_parallel_runtime = approximate_marginals_func(problem_name=problem_name, repeats=100, f=.03, m=5)
    
    #print approximate_marginals
    #unique, counts = np.unique(approximate_marginals, return_counts=True)
    #count_dict = dict(zip(unique, counts))
    #marginalCount_0_1 = 0
    #if 0.0 in count_dict:
    #    marginalCount_0_1 += count_dict[0.0]
    #if 1.0 in count_dict:
    #    marginalCount_0_1 += count_dict[1.0]
    #print 'number of 0 or 1 approximate_marginals =', marginalCount_0_1
    #exit(0)    
    #exit(0)


    restricting_hypercube = get_restricting_hypercube(approximate_marginals)
    print 'len(restricting_hypercube):', len(restricting_hypercube)
    #print 'restricting_hypercube1:', restricting_hypercube
    #exit(0)

    inside_hypercube_times = []
    outside_hypercube_times = []

    inside_hypercube_SAT_times = []
    outside_hypercube_SAT_times = []

    total_satisfied_sol_found_hypercube = 0
    total_satisfied_sol_found_outside_hypercube = 0
    print 'iter:',
    for i in range(REPEATS):
        print i,
        #print '#'*80
        #print 'iter:', i
        sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)

        hypercube_SAT_problem, outside_hypercube_SAT_problem, restricting_hypercube = sat.construct_smaller_SAT_problem(restricting_hypercube)
        hypercube_SAT_problem.add_parity_constraints(m=m, f=f)

        (outcome, empirical_density, satisfying_solution) = hypercube_SAT_problem.solve(3600, return_satisfying_solution=True)
        elapsed_time = outcome[1]
        inside_hypercube_times.append(elapsed_time)

        if outcome is True:
            solution = "true"
        elif outcome is False:
            solution = "false"
        else:
            solution = "timeout"
    
        if outcome[0] == True:
            total_satisfied_sol_found_hypercube += 1
            inside_hypercube_SAT_times.append(elapsed_time)

        #print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
        #print("problem_name: %s, solution=%s" % (problem_name, outcome))

        SOLVE_OUTSIDE_HYPERCUBE = True
        if SOLVE_OUTSIDE_HYPERCUBE:
            outside_hypercube_SAT_problem.add_parity_constraints(m=m, f=f)

            (outcome, empirical_density, satisfying_solution) = outside_hypercube_SAT_problem.solve(3600, return_satisfying_solution=True)
            elapsed_time = outcome[1]
            outside_hypercube_times.append(elapsed_time)

            if outcome is True:
                solution = "true"
            elif outcome is False:
                solution = "false"
            else:
                solution = "timeout"
        
            if outcome[0] == True:
                total_satisfied_sol_found_outside_hypercube += 1
                outside_hypercube_SAT_times.append(elapsed_time)

            #print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
            #print("problem_name: %s, solution=%s" % (problem_name, outcome))        

    print '^'*80
    print 'new problem, in out hypercube info:'
    print "mean_time_hypercube =", sum(inside_hypercube_times)/REPEATS
    print 'max_time_hypercube =', max(inside_hypercube_times)
    print 'max_time_SAT_hypercube =', max(inside_hypercube_SAT_times)

    print "mean_time_outside_hypercube =", sum(outside_hypercube_times)/REPEATS

    print "total_satisfied_sol_found_hypercube =", total_satisfied_sol_found_hypercube
    print "total_satisfied_sol_found_outside_hypercube =", total_satisfied_sol_found_outside_hypercube

def get_restricting_hypercube(approximate_marginals):  
    '''

    Inputs:
    - approximate_marginals: (list of floats) approximate marginals

    Outputs:
    - restricting_hypercube: (dictionary) specification of the hypercube.  Each entry 
        specifies a dimension to cut in half:
            -key: (int) 1 indexed dimension
            -value: (0 or 1) value of dimension in the hypercube

    '''
    restricting_hypercube = {}
    for var_idx, marginal in enumerate(approximate_marginals):
        if marginal == 1.0:
            restricting_hypercube[var_idx + 1] = 1
        elif marginal == 0.0:
            restricting_hypercube[var_idx + 1] = 0
    return restricting_hypercube

def test_bound_c880(problem_name='c880.isc', repeats=100, f=1.0001/417, m=60):
    sat = SAT("../../winter_2018/low_density_parity_checks/SAT_problems_cnf/%s" % problem_name, verbose=False, duplicate=0)    
    approximate_marginals, approximate_marginals_sequential_runtime,\
        approximate_marginals_parallel_runtime = approximate_marginals_func(problem_name=problem_name, repeats=100, f=1.0001/417, m=60, REGULAR=True, PERMUTE=False)
    

    def get_restricting_hypercube_c880(approximate_marginals):  
        '''
    
        Inputs:
        - approximate_marginals: (list of floats) approximate marginals
    
        Outputs:
        - restricting_hypercube: (dictionary) specification of the hypercube.  Each entry 
            specifies a dimension to cut in half:
                -key: (int) 1 indexed dimension
                -value: (0 or 1) value of dimension in the hypercube
    
        '''
        restricting_hypercube = {}
        for var_idx, marginal in enumerate(approximate_marginals):
            if var_idx < 60:
                continue
            elif marginal > .5:
                restricting_hypercube[var_idx + 1] = 1
            else:
                restricting_hypercube[var_idx + 1] = 0
        return restricting_hypercube
    
    
    restricting_hypercube = get_restricting_hypercube_c880(approximate_marginals)

    print 'len(restricting_hypercube):', len(restricting_hypercube)
    #print 'restricting_hypercube1:', restricting_hypercube
    #exit(0)

    inside_hypercube_times = []
    outside_hypercube_times = []

    inside_hypercube_SAT_times = []
    outside_hypercube_SAT_times = []

    total_satisfied_sol_found_hypercube = 0
    total_satisfied_sol_found_outside_hypercube = 0
    print 'iter:',
    for i in range(REPEATS):
        print i,
        #print '#'*80
        #print 'iter:', i
        sat = SAT("../../winter_2018/low_density_parity_checks/SAT_problems_cnf/%s" % problem_name, verbose=False, duplicate=0)

        hypercube_SAT_problem, outside_hypercube_SAT_problem, restricting_hypercube = sat.construct_smaller_SAT_problem(restricting_hypercube)
        hypercube_SAT_problem.add_regular_constraints_constantF_permuted(m=60, f=0.0, f_block=1, permute=False, k=1, ADD_CONSTRAINT_ALL_ONES=False, change_var_names=False)

        (outcome, empirical_density, satisfying_solution) = hypercube_SAT_problem.solve(3600, return_satisfying_solution=True)
        elapsed_time = outcome[1]
        inside_hypercube_times.append(elapsed_time)

        if outcome is True:
            solution = "true"
        elif outcome is False:
            solution = "false"
        else:
            solution = "timeout"
    
        if outcome[0] == True:
            total_satisfied_sol_found_hypercube += 1
            inside_hypercube_SAT_times.append(elapsed_time)

        #print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
        #print("problem_name: %s, solution=%s" % (problem_name, outcome))

        SOLVE_OUTSIDE_HYPERCUBE = True
        if SOLVE_OUTSIDE_HYPERCUBE:
            outside_hypercube_SAT_problem.add_parity_constraints(m=20, f=.05)

            (outcome, empirical_density, satisfying_solution) = outside_hypercube_SAT_problem.solve(3600, return_satisfying_solution=True)
            elapsed_time = outcome[1]
            outside_hypercube_times.append(elapsed_time)

            if outcome is True:
                solution = "true"
            elif outcome is False:
                solution = "false"
            else:
                solution = "timeout"
        
            if outcome[0] == True:
                total_satisfied_sol_found_outside_hypercube += 1
                outside_hypercube_SAT_times.append(elapsed_time)

            #print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
            #print("problem_name: %s, solution=%s" % (problem_name, outcome))        

    print '^'*80
    print 'new problem, in out hypercube info:'
    print "mean_time_hypercube =", sum(inside_hypercube_times)/REPEATS
    print 'max_time_hypercube =', max(inside_hypercube_times)
    print 'max_time_SAT_hypercube =', max(inside_hypercube_SAT_times)

    print "mean_time_outside_hypercube =", sum(outside_hypercube_times)/REPEATS

    print "total_satisfied_sol_found_hypercube =", total_satisfied_sol_found_hypercube
    print "total_satisfied_sol_found_outside_hypercube =", total_satisfied_sol_found_outside_hypercube


def test_halfspace_constraints(dimensions_to_cut, problem_name='c880.isc', m=60, f=.05, REPEATS=100, USE_PARITY_CONSTRAINTS=True):
    sat = SAT("../../winter_2018/low_density_parity_checks/SAT_problems_cnf/%s" % problem_name, verbose=False, duplicate=0)        
    all_runtimes = []
    SAT_runtimes = []
    total_satisfied_sol_found = 0
       
    for i in range(REPEATS):
        print i,
        sat = SAT("../../winter_2018/low_density_parity_checks/SAT_problems_cnf/%s" % problem_name, verbose=False, duplicate=0)

        
        if USE_PARITY_CONSTRAINTS:
            sat.add_halfspace_constraints(m=len(dimensions_to_cut), dimensions_to_cut=dimensions_to_cut)
        else: #make a new SAT problem, with fewer variables, that encodes half space constraints
            rand_restricting_hypercube = {}
            for var_idx in dimensions_to_cut:
                if random.randint(0, 1) == 0:
                    rand_restricting_hypercube[var_idx] = 0
                else:
                    rand_restricting_hypercube[var_idx] = 1

            new_SAT_problem, outside_hypercube_SAT_problem, restricting_hypercube = sat.construct_smaller_SAT_problem(rand_restricting_hypercube)
            sat = new_SAT_problem

        if m > len(dimensions_to_cut):
            remaining_m = m - len(dimensions_to_cut)
            variable_probs = [1.0 for i in range(sat.n)]
            for var_idx in range(1,sat.n+1):
                if var_idx in dimensions_to_cut:
                    variable_probs[var_idx-1] = 0.0
            variable_probs = np.array(variable_probs)
            variable_probs /= np.sum(variable_probs)
            sat.add_parity_constraints_proportional_to_marginals(m=remaining_m, f=f, marginals=None, variable_probs=variable_probs)

        (outcome, empirical_density, satisfying_solution) = sat.solve(3600, return_satisfying_solution=True)
        elapsed_time = outcome[1]
        all_runtimes.append(elapsed_time)
        print "satisfiable:", outcome[0], 'runtime:', outcome[1]
        if outcome[0] == True:
            total_satisfied_sol_found += 1
            SAT_runtimes.append(elapsed_time)

    print "mean_time =", sum(all_runtimes)/REPEATS
    #print "max_time =", max(all_runtimes)
    #print "max_SAT_time =", max(SAT_runtimes)
    print "total_satisfied_sol_found =", total_satisfied_sol_found


def test_SAT_solution_correlation(dimensions_to_cut, problem_name='c880.isc', m=60, REPEATS=100, USE_CACHE=True, VERBOSE=1):
    sat = SAT("../../winter_2018/low_density_parity_checks/SAT_problems_cnf/%s" % problem_name, verbose=False, duplicate=0)        
    all_runtimes = []
    SAT_runtimes = []
    total_satisfied_sol_found = 0
       
    new_satisfying_solutions = []

    for i in range(REPEATS):
        print i,
        sat = SAT("../../winter_2018/low_density_parity_checks/SAT_problems_cnf/%s" % problem_name, verbose=False, duplicate=0)

        sat.add_halfspace_constraints(m=m, dimensions_to_cut=dimensions_to_cut)

        (outcome, empirical_density, satisfying_solution) = sat.solve(3600, return_satisfying_solution=True)

        if satisfying_solution:
            new_satisfying_solutions.append(np.array(satisfying_solution))

        elapsed_time = outcome[1]
        all_runtimes.append(elapsed_time)
        print "satisfiable:", outcome[0], 'runtime:', outcome[1]
        if outcome[0] == True:
            total_satisfied_sol_found += 1
            SAT_runtimes.append(elapsed_time)

    if USE_CACHE:
        if not os.path.isdir("./cached_samples"):
            os.mkdir("./cached_samples")
        cache_file_location = './cached_samples/%s_satisfyingSolutionSamples.cache' % problem_name
        try:
            with open(cache_file_location, 'rb') as cache_file:
                print 'opened'
                previously_computed_satisfying_solutions = pickle.load(cache_file)
        except IOError:
            print 'did not open'
            previously_computed_satisfying_solutions = []
        all_satisfying_solutions = new_satisfying_solutions + previously_computed_satisfying_solutions #+ used for list concatenation
        with open(cache_file_location, 'wb') as cache_file: #would be faster to write on the end rather than rewriting everything if this matters
            pickle.dump(all_satisfying_solutions, cache_file)
    else:
        all_satisfying_solutions = new_satisfying_solutions

    all_satisfying_solutions=np.asarray(all_satisfying_solutions)

    print "number of satisfying solutions sampled =", len(all_satisfying_solutions)
    covariance = np.cov(all_satisfying_solutions, rowvar=False)
    print 'hi1'
    np.set_printoptions(precision=2)
    if VERBOSE == 1:
        print 'hi2'
        marginals = np.mean(all_satisfying_solutions, axis=0)
        max_diff_from_half = 0
        max_cov = 0
        for var in range(m):
            cur_diff_from_half = abs(.5 - marginals[var])
            if cur_diff_from_half > max_diff_from_half:
                max_diff_from_half = cur_diff_from_half

            for var2 in range(m):
                if var2 != var:
                    cur_cov = np.abs(covariance[var,var2])
                    if cur_cov > max_cov:
                        max_cov = cur_cov

        for var in range(m, covariance.shape[0]):
            cur_diff_from_half = abs(.5 - marginals[var])
            max_first_60_cov = 0
            for var2 in range(m):
                cur_cov = np.abs(covariance[var,var2])
                if cur_cov > max_first_60_cov:
                    max_first_60_cov = cur_cov
            if cur_diff_from_half < 3*max_diff_from_half and max_first_60_cov < 3*max_cov:
                print 'var =', var, 'marginal:', np.mean(all_satisfying_solutions, axis=0)[var], ', top 5 covariance values with variables 1 to m:', ["{0:0.2f}".format(i) for i in sorted(np.abs(covariance[var,0:m]), reverse=True)[0:5]]                

    if VERBOSE == 2:
        print covariance
        print 'covariance for variable 1', covariance[0,:]
        print 'covariance for variable 2', covariance[1,:]
        print 'covariance for variable 3', covariance[2,:]
        print 'covariance for variable 4', covariance[3,:]
        print 'sorted indices:', np.argsort(covariance[0,:])
        print 'top 5:', sorted(covariance[0,:])[0:5]
    
        for var_idx in range(covariance.shape[0]):
            print 'var =', var_idx, 'marginal:', np.mean(all_satisfying_solutions, axis=0)[var_idx], ', top 5 covariance values with variables 1 to m:', ["{0:0.2f}".format(i) for i in sorted(np.abs(covariance[var_idx,0:m]), reverse=True)[0:5]]
            print ["{0:0.2f}".format(i) for i in covariance[var_idx,0:m]]
    
        print "mean_time =", sum(all_runtimes)/REPEATS
        #print "max_time =", max(all_runtimes)
        #print "max_SAT_time =", max(SAT_runtimes)
        print "total_satisfied_sol_found =", total_satisfied_sol_found
    
        print 'var =', m-1, 'marginal:', np.mean(all_satisfying_solutions, axis=0)[m-1], ', top 5 covariance values with variables 1 to m:', ["{0:0.2f}".format(i) for i in covariance[m-1,0:m]]

    if VERBOSE == 3:
        all_pairwise_mutual_info = np.zeros(covariance.shape)
        for var in range(covariance.shape[0]):
            print 'var=', var
            #assert(len(all_satisfying_solutions[:, var]) == REPEATS)
            mutual_info = sklearn.feature_selection.mutual_info_classif(all_satisfying_solutions, all_satisfying_solutions[:, var], discrete_features=True)
            for i in range(covariance.shape[0]):
                if i < var:
                    assert(abs(all_pairwise_mutual_info[i, var] - mutual_info[i]) < .0001), (all_pairwise_mutual_info[i, var], mutual_info[i], var, i)
                    print '(all_pairwise_mutual_info[i, var] = ', all_pairwise_mutual_info[i, var]
                    print 'mutual_info[i] = ', mutual_info[i]
                all_pairwise_mutual_info[var, i] = mutual_info[i]
                
                

                #print all_pairwise_mutual_info[:8, :8]
    
        if USE_CACHE:
            mutual_info_matrix_file = './cached_samples/%s_mutualInformationMatrix.pickle' % problem_name
            #try:
            #    with open(mutual_info_matrix_file, 'rb') as cache_file:
            #        print 'opened'
            #        all_pairwise_mutual_info = pickle.load(cache_file)
            #except IOError:
            #    print 'did not open'
            with open(mutual_info_matrix_file, 'wb') as cache_file: #would be faster to write on the end rather than rewriting everything if this matters
                pickle.dump(all_pairwise_mutual_info, cache_file)    

        for var_idx in range(covariance.shape[0]):
            print 'var =', var_idx, 'marginal:', np.mean(all_satisfying_solutions, axis=0)[var_idx], ', top 5 mutual_info values with variables 1 to m:', ["{0:0.2f}".format(i) for i in sorted(np.abs(all_pairwise_mutual_info[var_idx,0:m]), reverse=True)[0:5]]




def get_variable_subset(problem_name):
    filename = '/atlas/u/jkuck/mis/mis_results/%s.out' % problem_name.split('.')[0]

    ifstream = open(filename)
    line = ifstream.readline().split()
    assert(line[0] == 'v' and line[-1] == '0')
    variables = line[1:-1]
    ifstream.close()
    variables_as_ints = []
    for var in variables:
        variables_as_ints.append(int(var))
    return variables_as_ints

def test_variable_subset(problem_name='log-1.cnf'):
    m = 70
    f = .04
    REPEATS = 100
    #VARIABLE_SUBSET = []
    VARIABLE_SUBSET = get_variable_subset(problem_name)
    #sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)
    #VARIABLE_SUBSET = [i for i in range(1, sat.n+1) if not(i in VARIABLE_SUBSET)]

    all_runtimes = []
    all_runtimes_debug = []
    SAT_runtimes = []
    SAT_runtimes_debug = []
    total_satisfied_sol_found = 0
    total_satisfied_sol_found_debug = 0

    for i in range(REPEATS):
        #print '#'*80
        #print 'iter:', i
        sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)

        REGULAR = True

        if REGULAR:
            if VARIABLE_SUBSET == []:
                VARIABLE_SUBSET=[i+1 for i in range(sat.n)]

            k = np.floor(len(VARIABLE_SUBSET)/m)
            k_density = k/len(VARIABLE_SUBSET)
            if k_density > f: #we need to decrease k
                k = np.floor(f*len(VARIABLE_SUBSET))

            f_prime = (f*len(VARIABLE_SUBSET) - k)/(len(VARIABLE_SUBSET) - 2*k)
            print 'f_prime:', f_prime, 'k=', k, 'len(VARIABLE_SUBSET)=', len(VARIABLE_SUBSET)

            sat.add_regular_constraints_subsetOfVariables(m=m, f=f_prime, f_block=1-f_prime, permute=True, k=k, ADD_CONSTRAINT_ALL_ONES=False, change_var_names=False, variable_subset=VARIABLE_SUBSET)
      
            #SAT_problem_debug = sat.add_regular_constraints_constantF_permuted_debug(m=m, f=f_prime, f_block=1-f_prime, permute=True, k=k, ADD_CONSTRAINT_ALL_ONES=False, change_var_names=False, variable_subset=VARIABLE_SUBSET)
        else:
            sat.add_regular_constraints_subsetOfVariables(m=m, f=f, f_block=0, permute=False, k=0, ADD_CONSTRAINT_ALL_ONES=False, change_var_names=False, variable_subset=VARIABLE_SUBSET)

        (outcome, empirical_density, satisfying_solution) = sat.solve(3600, return_satisfying_solution=True)
        #(outcome_debug, empirical_density_debug, satisfying_solution_debug) = SAT_problem_debug.solve(3600, return_satisfying_solution=True)
        #assert(outcome[0] == outcome_debug[0])

        #print "satisfying_solution:", satisfying_solution
        elapsed_time = outcome[1]
        all_runtimes.append(elapsed_time)
        print "satisfiable:", outcome[0], 'runtime:', outcome[1]
        if outcome[0] == True:
            total_satisfied_sol_found += 1
            SAT_runtimes.append(elapsed_time)

        #all_runtimes_debug.append(outcome_debug[1])
        #if outcome_debug[0] == True:
        #    total_satisfied_sol_found_debug += 1
        #    SAT_runtimes.append(elapsed_time)

    print '-'*30, 'ALL VARIABLES', '-'*30
    print "mean_time =", np.mean(all_runtimes)
    print "median_time =", np.median(all_runtimes)
    print "min_time =", np.min(all_runtimes)
    print "max_time =", np.max(all_runtimes)
    #print "max_time =", max(all_runtimes)
    #print "max_SAT_time =", max(SAT_runtimes)
    print "total_satisfied_sol_found =", total_satisfied_sol_found
    print 'len(VARIABLE_SUBSET) =', len(VARIABLE_SUBSET)
    print VARIABLE_SUBSET

    print '-'*30, 'SUBSET OF VARIABLES', '-'*30
    print "mean_time =", np.mean(all_runtimes_debug)
    print "median_time =", np.median(all_runtimes_debug)
    print "min_time =", np.min(all_runtimes_debug)
    print "max_time =", np.max(all_runtimes_debug)

    print "total_satisfied_sol_found =", total_satisfied_sol_found_debug
    print 'len(VARIABLE_SUBSET) =', len(VARIABLE_SUBSET)



if __name__ == '__main__':
#############    dimensions_to_cut = [i for i in range(1,37)] 
#############    test_SAT_solution_correlation(dimensions_to_cut, problem_name='c432.isc', m=36, REPEATS=0, USE_CACHE=True, VERBOSE=1)
#############    exit(0)
#########    dimensions_to_cut = [i for i in range(1,50)] #+ [343]
#########    test_halfspace_constraints(dimensions_to_cut=dimensions_to_cut, problem_name='c2670.isc', m=51, f=.5, REPEATS=100, USE_PARITY_CONSTRAINTS=True)
#############
#############    #dimensions_to_cut = [i for i in range(1,236)]
#############    #test_halfspace_constraints(dimensions_to_cut=dimensions_to_cut, problem_name='c2670.isc', m=len(dimensions_to_cut), REPEATS=100)
#############
#############    dimensions_to_cut = [i for i in range(1,5)]
#############    test_halfspace_constraints(dimensions_to_cut=dimensions_to_cut, problem_name='log-2.cnf', m=len(dimensions_to_cut), REPEATS=100)
#############
#########    exit(0)

    m = 40#33
    f = 0.05#0.05    
    #print get_variable_subset('log-1.cnf')
    test_variable_subset()
    exit(0)
    #m = 20#33
    #f = 0.03#0.05

    m = 60
    f = .003
    f_block = 1
    REPEATS = 100
    all_runtimes = []
    SAT_runtimes = []
    total_satisfied_sol_found = 0
#    PROBLEM_NAMES = ['c432.isc', 'c499.isc', 'c880.isc', 'c1355.isc', 'c1908.isc', 'c2670.isc', 'sat-grid-pbl-0010.cnf', 'sat-grid-pbl-0015.cnf', 'sat-grid-pbl-0020.cnf', 'ra.cnf', 'tire-1.cnf', 'tire-2.cnf', 'tire-3.cnf', 'tire-4.cnf', 'log-1.cnf', 'log-2.cnf', 'lang12.cnf']
#    PROBLEM_NAMES = ['sat-grid-pbl-0010.cnf']
#    PROBLEM_NAMES = ['tire-4.cnf']
    PROBLEM_NAMES = ['c880.isc']
    USE_MARGINALS = True

    #test_bound_c880()
    #print approximate_marginals_func(problem_name=PROBLEM_NAMES[0], repeats=100, f=1.0001/417, m=60, REGULAR=True, PERMUTE=False)
    #solve_smaller_sat(problem_name=PROBLEM_NAMES[0], repeats=REPEATS, f=f, m=m)
    #exit(0)
    for problem_name in PROBLEM_NAMES:
        sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)        
        if USE_MARGINALS:
            approximate_marginals, approximate_marginals_sequential_runtime, approximate_marginals_parallel_runtime = approximate_marginals_func(problem_name=problem_name, repeats=100, f=.07, m=10)
            #approximate_marginals = np.array([1 for i in range(50)] + [.5 for i in range(50)]) #for hypercube1
            print '@'*80
            print 'approximate_marginals:', approximate_marginals
            restricting_hypercube = get_restricting_hypercube(approximate_marginals)
            f *= (sat.n-len(restricting_hypercube))/sat.n
            print 'f=', f, 'sat.n=', sat.n, 'len(restricting_hypercube)=', len(restricting_hypercube)
        exit(0)
        for i in range(REPEATS):
            print i,
            #print '#'*80
            #print 'iter:', i
            sat = SAT("%s/%s" % (SAT_PROBLEM_FOLDER,problem_name), verbose=False, duplicate=0)

            REGULAR = True

            if REGULAR:
                k = np.floor(sat.n/m)
                k_density = k/sat.n
                if k_density > f: #we need to decrease k
                    k = np.floor(f*sat.n)
    
                f_prime = (f*sat.n - k)/(sat.n - 2*k)
                print 'f_prime:', f_prime, 'k=', k, 'n=', sat.n
                sat.add_regular_constraints_constantF_permuted(m=m, f=f_prime, f_block=1-f_prime, permute=False, k=k, ADD_CONSTRAINT_ALL_ONES=False, change_var_names=False)
          
            else:
                if USE_MARGINALS:
                    sat.add_parity_constraints_proportional_to_marginals(m=m, f=f, marginals=approximate_marginals)
                else:
                    sat.add_parity_constraints_restrict_vars(m=m, f=f, var_restriction=int(sat.n/1))
                    #sat.add_parity_constraints(m=m, f=f)

            #sat.add_permutation_constraints(m=sat.n, f=0.05)
            #sat.add_variable_length_constraints(m=sat.n, f=0.05)

            #sat.add_permutation_constraints(m=sat.n, f=0.02, ADD_CONSTRAINT_ALL_ONES=False)
            #sat.add_permutation_constraints(m=sat.n, f=0.02, ADD_CONSTRAINT_ALL_ONES=True)
            
            #sat.add_parity_constraints(m=sat.n, f=.0590909)

            #sat.add_regular_constraints_constantF_permuted(m=m, f=f, f_block=f_block)
            #sat.add_regular_constraints_constantF(m, f)
            (outcome, empirical_density, satisfying_solution) = sat.solve(3600, return_satisfying_solution=True)
            #print "satisfying_solution:", satisfying_solution
            elapsed_time = outcome[1]
            all_runtimes.append(elapsed_time)
            if outcome is True:
                solution = "true"
            elif outcome is False:
                solution = "false"
            else:
                solution = "timeout"
        
            print "satisfiable:", outcome[0], 'runtime:', outcome[1]
            if outcome[0] == True:
                total_satisfied_sol_found += 1
                SAT_runtimes.append(elapsed_time)
            #print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
            #print("problem_name: %s, solution=%s" % (problem_name, outcome))

        print "mean_time =", sum(all_runtimes)/REPEATS
        #print "max_time =", max(all_runtimes)
        #print "max_SAT_time =", max(SAT_runtimes)
        print "total_satisfied_sol_found =", total_satisfied_sol_found
        if USE_MARGINALS:
            print 'approximate_marginals_sequential_runtime:',approximate_marginals_sequential_runtime
            print 'approximate_marginals_parallel_runtime:', approximate_marginals_parallel_runtime
        exit(0)















    M = 19
    K = 5
    problem_instance = "lang12"

    block_diag_f_vals = []
    solve_times_orig = []
    solve_times_orig_f_prime = []
    solve_times_block_diag = []
    f_min = .01
    f_max = .11
    f_step = .01
    f = f_min
    REPEATS = 20
    while f <= f_max:
#        sat_orig = read_SAT_problem("SAT_problems_cnf/%s" % "sat-grid-pbl-0010.cnf")
#        sat_orig.add_parity_constraints(m = M, f=f, use_XOR_clauses=True)
#        solution, time_orig = solve_SAT(sat_orig)
#        solve_times_orig.append(time_orig)
#
#        sat_orig_f_prime = read_SAT_problem("SAT_problems_cnf/%s" % "sat-grid-pbl-0010.cnf")
#        sat_orig_f_prime.add_parity_constraints(m = M, f=f+K/N, use_XOR_clauses=True)
#        solution, time_orig_f_prime = solve_SAT(sat_orig_f_prime)
#        solve_times_orig_f_prime.append(time_orig_f_prime)
#      
#        sat_block_diag = read_SAT_problem("SAT_problems_cnf/%s" % "sat-grid-pbl-0010.cnf")
#        sat_block_diag.add_parity_constraints_block_diagonal(m = M, k=K, f=f, use_XOR_clauses=True)    
#        solution, time_block_diag = solve_SAT(sat_block_diag)
#        solve_times_block_diag.append(time_block_diag) 

        for itr in range(REPEATS):


            problem = SAT('examples/%s.cnf'%problem_instance, verbose=True)
            problem.n = 100
            problem.add_parity_constraints(m=M, f=f)
            ((satisfied_orig, run_time_orig), empirical_density) = problem.solve()

            problem = SAT('examples/%s.cnf'%problem_instance, verbose=True)
            problem.n = 100
            problem.add_regular_constraints(m=M, f=f)
            ((satisfied_regular, run_time_regular), empirical_density) = problem.solve()

            problem = SAT('examples/%s.cnf'%problem_instance, verbose=True)
            problem.n = 100
            N = problem.n
            problem.add_parity_constraints(m=M, f=f+K/N)
            ((satisfied_orig_fprime, run_time_orig_fprime), empirical_density) = problem.solve()

            solve_times_orig.append(run_time_orig)
            solve_times_block_diag.append(run_time_regular) 
            solve_times_orig_f_prime.append(run_time_orig_fprime)
    
            block_diag_f_vals.append(f)

        f+=f_step

    fig = plt.figure()
    ax = plt.subplot(111)

    ax.scatter(block_diag_f_vals, solve_times_orig, c = 'b', marker='+', label='orig f, mean=%f, median=%f' % (np.mean(solve_times_orig), np.median(solve_times_orig)))
    ax.scatter([f - .0004 for f in block_diag_f_vals], solve_times_orig_f_prime, c = 'r', marker='x', label='orig f_prime=f+k/n, mean=%f, median=%f' % (np.mean(solve_times_orig_f_prime), np.median(solve_times_orig_f_prime)))
    ax.scatter([f + .0004 for f in block_diag_f_vals], solve_times_block_diag, c='g', marker='_', label="block diag f, mean=%f, median=%f"%(np.mean(solve_times_block_diag), np.median(solve_times_block_diag)))
    #ax.set_yscale('log')


    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig('./new_plots/%s_M=%d_k=%d_dependent_SAT_solver_time_f=%d-%d' % (problem_instance, M, K, f_min/f_step, f_max/f_step), bbox_extra_artists=(lgd,), bbox_inches='tight')    
    plt.close()    # close the figure

    sleep(1)



    problem = SAT('examples/lang12.cnf', verbose=True)
#    problem.n = 100
    problem.add_parity_constraints(m=20, f=.05)
    problem.solve()
    exit(0)

    problem = SAT('examples/lang12.cnf', verbose=True)
    problem.n = 100
    problem.add_regular_constraints(12, 0.02)
    problem.solve()
    exit(0)

    true_count = 0
    false_count = 0
    m = 27
    for i in range(0, 1000):
        problem.add_regular_constraints(m, 0.02)
        (result, empirical_density) = problem.solve()
        if result is True:
            print(str(m) + " SAT: " + str(true_count) + ":" + str(false_count))
            true_count += 1
        elif result is False:
            print(str(m) + " UNSAT" + str(true_count) + ":" + str(false_count))
            false_count += 1

