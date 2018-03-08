__author__ = 'shengjia'

import subprocess, threading
import random
import time
import os
import math

MACHINE = 'atlas' #'atlas' or 'local'

if MACHINE == 'local':
    CRYPTOMINISAT5_DIRECTORY = '/Users/jkuck/software/cryptominisat-5.0.1/build'
    SAT_SOLVER = "CRYPTOMINISAT5"
else:
    import matplotlib
    matplotlib.use('Agg') #prevent error running remotely
    import matplotlib.pyplot as plt
    import numpy as np
    INSTALL_DIRECTORY = '/atlas/u/jkuck'

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

    def add_parity_constraints(self, m, f):
        """ Add m parity constraints, each atom is included into the xor with probability f """
        if self.duplicate_factor == 1:
            self.hash_functions = []
            self.new_variables = 0
            self.fail_apriori = False
    
            cur_index = self.n + 1
    
            print "n = ", self.n
            print "clauseCount = ", self.clauseCount
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
    
            print "n = ", self.n
            print "clauseCount = ", self.clauseCount
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

    def add_regular_constraints_constantF_permuted(self, m, f, f_block, permute=True, k=None, ADD_CONSTRAINT_ALL_ONES=False):
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
        elif k == 'maxConstant':
            k_low = int(math.floor(float(self.n) / m_effective))
            k_high = int(math.floor(float(self.n) / m_effective))            
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
        if permute:
            permuted_block_diag_matrix = np.swapaxes(np.random.permutation(np.swapaxes(block_diag_matrix,0,1)),0,1)
        else:
            permuted_block_diag_matrix = block_diag_matrix
#        print permuted_block_diag_matrix
        f_updated = f 

        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        curIndex = self.n + 1

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
                    if random.random() < f_block:
                        new_function.append(atom)
                #if this element isn't part of a permuted block, add variable with probability f_updated
                elif random.random() < f_updated:
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


    def solve(self, max_time=-1):
        print 'hi'
        """ Attempt to solve the problem, returns True/False if the problem is satisfiable/unsatisfiable,
    returns None if timeout """
        if self.fail_apriori:
            if self.verbose:
                print("Constraint with zero length generated and inconsistent, UNSAT")
            return (False, 0)

        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        filename = "tmp/SAT_" + str(self.id) + ".cnf"
        ofstream = open(filename, "w")
        ofstream.write("p cnf " + str(self.n + self.new_variables) + " " + str(len(self.clauses) + len(self.hash_functions)) + "\n")

        for item in self.clauses:
            ofstream.write(item + "\n")
        for hashFunction in self.hash_functions:
            ofstream.write("x")
            for item in hashFunction:
                ofstream.write(str(item) + " ")
            ofstream.write("0\n")
        ofstream.close()
        start_time = time.time()
        if MACHINE == 'local':
            if SAT_SOLVER == 'ORIGINAL':
                solver = Command(['./cryptominisat', '--verbosity=0', '--gaussuntil=400', '--threads=1', filename])
            elif SAT_SOLVER == 'CRYPTOMINISAT5':
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
                print("-----------------------------------------")

        if result is None:
            return None

        result = result.split()
        if len(result) >= 2:
            outcome = result[1]
            if outcome == 'SATISFIABLE':
                return (True, end_time - start_time)
            elif outcome == 'UNSATISFIABLE':
                return (False, end_time - start_time)

        print("Error: unrecognized return value")
        print("Full output: ")
        print(result)
        return None


if __name__ == '__main__':
    #m = 20#33
    #f = 0.03#0.05

    m = 9
    #f = 0.052727
    #f = 0.05
    f = 0.02
    f_block = 1
    REPEATS = 10
    total_time = 0.0
    total_satisfied_sol_found = 0
#    PROBLEM_NAMES = ['c432.isc', 'c499.isc', 'c880.isc', 'c1355.isc', 'c1908.isc', 'c2670.isc', 'sat-grid-pbl-0010.cnf', 'sat-grid-pbl-0015.cnf', 'sat-grid-pbl-0020.cnf', 'ra.cnf', 'tire-1.cnf', 'tire-2.cnf', 'tire-3.cnf', 'tire-4.cnf', 'log-1.cnf', 'log-2.cnf', 'lang12.cnf']
    PROBLEM_NAMES = ['sat-grid-pbl-0010.cnf']
    for problem_name in PROBLEM_NAMES:
        for i in range(REPEATS):
            sat = SAT("../../low_density_parity_checks/SAT_problems_cnf/%s" % problem_name, verbose=False, duplicate=0)
            sat.add_regular_constraints_constantF_permuted(m=5, f=.01, f_block=1, permute=True, k=None, ADD_CONSTRAINT_ALL_ONES=True)
            #sat.add_permutation_constraints(m=sat.n, f=0.05)
            #sat.add_variable_length_constraints(m=sat.n, f=0.05)

            #sat.add_permutation_constraints(m=sat.n, f=0.02, ADD_CONSTRAINT_ALL_ONES=False)
            #sat.add_permutation_constraints(m=sat.n, f=0.02, ADD_CONSTRAINT_ALL_ONES=True)
            
            #sat.add_parity_constraints(m=sat.n, f=.0590909)

            sat.add_regular_constraints_constantF_permuted(m=m, f=f, f_block=f_block)
            #sat.add_regular_constraints_constantF(m, f)
            outcome = sat.solve(3600)
            elapsed_time = outcome[1]
            total_time += elapsed_time
            if outcome is True:
                solution = "true"
            elif outcome is False:
                solution = "false"
            else:
                solution = "timeout"
        
            if outcome[0] == True:
                total_satisfied_sol_found += 1
            #print("f = %.3f, time=%.2f, solution=%s" % (f, elapsed_time, outcome))
            print("problem_name: %s, solution=%s" % (problem_name, outcome))

    print "mean_time =", total_time/REPEATS
    print "total_satisfied_sol_found =", total_satisfied_sol_found
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
            (satisfied_orig, run_time_orig) = problem.solve()

            problem = SAT('examples/%s.cnf'%problem_instance, verbose=True)
            problem.n = 100
            problem.add_regular_constraints(m=M, f=f)
            (satisfied_regular, run_time_regular) = problem.solve()

            problem = SAT('examples/%s.cnf'%problem_instance, verbose=True)
            problem.n = 100
            N = problem.n
            problem.add_parity_constraints(m=M, f=f+K/N)
            (satisfied_orig_fprime, run_time_orig_fprime) = problem.solve()

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
        result = problem.solve()
        if result is True:
            print(str(m) + " SAT: " + str(true_count) + ":" + str(false_count))
            true_count += 1
        elif result is False:
            print(str(m) + " UNSAT" + str(true_count) + ":" + str(false_count))
            false_count += 1

