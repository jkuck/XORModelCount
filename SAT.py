__author__ = 'shengjia'

import subprocess, threading
import random
import time
import os

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
        self.newVariables = 0

        if verbose:
            print("CNF with " + str(self.n) + " variables and " + str(self.clauseCount) + " clauses")

        self.verbose = verbose
        self.hashFunctions = []
        self.failApriori = False

    '''Add m parity constraints, each atom is included into the xor with probability f'''
    def parityConstraints(self, m, f):
        self.hashFunctions = []
        self.newVariables = 0
        self.failApriori = False

        curIndex = self.n + 1

        for i in range(0, m):
            new_function = []

            for atom in range(1, self.n + 1):
                if random.random() < f:
                    new_function.append(atom)
            if len(new_function) == 0:
                if random.randint(0, 1) == 0:
                    continue
                else:
                    self.failApriori = True
                    return
            if random.randint(0, 1) == 0:
                new_function[0] = -new_function[0]
            if self.max_xor > 0:
                while len(new_function) > self.max_xor:
                    temp = new_function[0:self.max_xor - 1]
                    new_function = [curIndex] + new_function[self.max_xor - 1:]
                    temp.append(curIndex)
                    curIndex += 1
                    self.newVariables += 1
                    self.hashFunctions.append(temp)
            self.hashFunctions.append(new_function)
        if self.verbose:
            print("Generated " + str(m) + " parity constraints")
            if self.max_xor > 0:
                print("Maximum xor length is " + str(self.max_xor) +
                      ". Added " + str(self.newVariables) + " new variables")

    ''' Attempt to solve the problem, returns True/False if the problem is satisfiable/unsatisfiable,
    returns None if timeout '''
    def solve(self, max_time=-1):
        if self.failApriori:
            if self.verbose:
                print("Constraint with zero length generated and inconsistent, UNSAT")
            return False

        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        filename = "tmp/SAT_" + str(self.id) + ".cnf"
        ofstream = open(filename, "w")
        ofstream.write("p cnf " + str(self.n + self.newVariables) + " " + str(len(self.clauses) + len(self.hashFunctions)) + "\n")

        for item in self.clauses:
            ofstream.write(item + "\n")
        for hashFunction in self.hashFunctions:
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


''' Only used for internal testing '''
if __name__ == '__main__':
    problem = SAT('dataset/lang15.cnf', verbose=False)
    true_count = 0
    false_count = 0
    m = 27
    for i in range(0, 1000):
        problem.parityConstraints(m, 0.02)
        result = problem.solve()
        if result is True:
            print(str(m) + " SAT: " + str(true_count) + ":" + str(false_count))
            true_count += 1
        elif result is False:
            print(str(m) + " UNSAT" + str(true_count) + ":" + str(false_count))
            false_count += 1

