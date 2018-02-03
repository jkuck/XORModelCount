__author__ = 'shengjia'

import subprocess, threading
import random
import time
import os
import math

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
    def __init__(self, problem_file, verbose=True, instance_id=-1):
        # This id must be unique to this SAT instance
        if instance_id == -1:
            self.id = time.time()
        else:
            self.id = instance_id

        # Read in the problem instance
        ifstream = open(problem_file)
        self.clauses = []
        header = ifstream.readline().split()
        self.n = int(header[2])

        while True:
            curline = ifstream.readline()
            if not curline:
                break
            self.clauses.append(curline.strip())
        self.clauseCount = len(self.clauses)
        if self.clauseCount != int(header[3]):
            print("Warning: clause count mismatch, expecting " + str(header[3]) + ", received " + str(self.clauseCount))

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
        self.hash_functions = []
        self.new_variables = 0
        self.fail_apriori = False

        cur_index = self.n + 1

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
                
    def solve(self, max_time=-1):
        """ Attempt to solve the problem, returns True/False if the problem is satisfiable/unsatisfiable,
    returns None if timeout """
        if self.fail_apriori:
            if self.verbose:
                print("Constraint with zero length generated and inconsistent, UNSAT")
            return False

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
        solver = Command(['./cryptominisat', '--verbosity=0', '--gaussuntil=400', '--threads=1', filename])
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
                return True
            elif outcome == 'UNSATISFIABLE':
                return False

        print("Error: unrecognized return value")
        print("Full output: ")
        print(result)
        return None


if __name__ == '__main__':
    problem = SAT('examples/lang12.cnf', verbose=False)
    problem.n = 100
    problem.add_regular_constraints(12, 0.02)
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

