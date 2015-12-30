__author__ = 'shengjia'

from bigfloat import BigFloat
import bigfloat as bf
import random
import math

''' Class that computes binomial and partial sum of binomials and returns values in BigFloat
    This class once computed a result, caches it, after which each query only takes O(1) time '''
class BigBinom:
    def __init__(self, n):
        self.n = n
        # binom_list[m] = binom(n, m)
        self.binom_list = []
        # partial_binom_sum[m] = \sum_{w = 0}^{m} binom(n, w)
        self.partial_binom_sum = [BigFloat(1)]

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


''' Generate a data set with clauses size drawn uniformly from atom_count_set '''
def gen_data_set(file_name, num_of_var, num_of_clause, atom_count_set):
    out_file = open(file_name, "w")
    out_file.write("p cnf " + str(num_of_var) + " " + str(num_of_clause) + "\n")
    for i in range(0, num_of_clause):
        atom_count = random.choice(atom_count_set)
        clause = []
        for j in range(0, atom_count):
            atom = random.randint(1, num_of_var)
            while atom in clause:
                atom = random.randint(1, num_of_var)
            if random.randint(0, 1) == 0:
                atom = -atom
            clause.append(atom)
            out_file.write(str(atom) + " ")
        out_file.write("0\n")
    out_file.close()


''' Internal testing '''
if __name__ == '__main__':
    # gen_data_set("SAT3.cnf", 200, 650, [3])
    for n_index in range(10, 20):
        n = 2 ** n_index
        BigBinom.factorial(n)
    for n_index in range(10, 20):
        n = 2 ** n_index
        BigBinom.factorial(n, 1000)

