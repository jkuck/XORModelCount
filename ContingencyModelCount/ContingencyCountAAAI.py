import sys
sys.path.append("/home/fs01/se237/ProdRank/build")
from z3 import *
import argparse
import time
from random import randint
#from xorLength import f_star
import random
import math
import scipy
import scipy.misc

def toSMT2Benchmark(f, status="unknown", name="benchmark", logic=""):
  v = (Ast * 0)()
  return Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())


def convertor(f, status="unknown", name="benchmark", logic=""):
	v = (Ast * 0)()
	if isinstance(f, Solver):
		a = f.assertions()
	if len(a) == 0:
		f = BoolVal(True)
	else:
		f = And(*a)
	return Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())

def mean(i,M):
	return pow(2,(i - M))
	
def w_star(L,i):
	j = 1
	tmp =  scipy.misc.comb(L,j)
	while (tmp<pow(2.0,i)-1):
		j = j +1
		tmp =tmp + scipy.misc.comb(L,j)
	#print  "wstar=",j, " tot=",tmp, " S=",pow(2.0,i)
	return j	
	
## upper bound on the power
def po_rw(L,i,M,t, verbose=False):
	tmp = 0.0
	tot_conf = 0.0
	wstar=w_star(L,i)
	if verbose:
		print "wstar=",wstar
	## fraction of 1 in the matrix
	pos_fract = t
	
	## this can even be row dependent..
	
	for w in range(1,wstar):
		base = 0.5 + 0.5*((1.0-2*pos_fract)**w)
		tmp = tmp + scipy.misc.comb(L,w)* (base** M)
		tot_conf += scipy.misc.comb(L,w)
		if verbose:
			print (w,(base** M))
			print scipy.misc.comb(L,w)
		#print  "power=",tmp
	
	## add rest at maximum distance
	w=wstar
	base = 0.5 + 0.5*((1.0-2*pos_fract)**w)
	if verbose:
		print (w,(base** M))
		print (pow(2.0,i)-tot_conf)
	tmp = tmp + (pow(2.0,i)-1-tot_conf)* (base** M)
	
	tmp = tmp *pow(2.0,i-M)
	tmp = tmp +pow(2.0,i-M)
	if verbose:
		print  "power=",tmp,"(",tot_conf,")"
	return tmp


## upper bound on the power
def po_rw_over_musquared(L,i,M,t, verbose=False):
	tmp = 0.0
	tot_conf = 0.0
	wstar=w_star(L,i)
	if verbose:
		print "wstar=",wstar
	## fraction of 1 in the matrix
	pos_fract = t
	
	## this can even be row dependent..
	
	for w in range(1,wstar):
		base = 0.5 + 0.5*((1.0-2*pos_fract)**w)
		tmp = tmp + scipy.misc.comb(L,w)* (base** M)
		tot_conf += scipy.misc.comb(L,w)
		if verbose:
			print (w,(base** M))
			print scipy.misc.comb(L,w)
		#print  "power=",tmp
	
	## add rest at maximum distance
	w=wstar
	base = 0.5 + 0.5*((1.0-2*pos_fract)**w)
	if verbose:
		print (w,(base** M))
		print (pow(2.0,i)-tot_conf)
	tmp = tmp + (pow(2.0,i)-1-tot_conf)* (base** M)
	
	tmp = tmp *pow(2.0,i-M)
	tmp = tmp +pow(2.0,i-M)
	if verbose:
		print  "power=",tmp,"(",tot_conf,")"
	return tmp
	
	
	
## variance	
def v(L,i,M,t):
	return po_rw(L,i,M,t) - mean(i,M)*mean(i,M)
	
def upper_bound_star(L,M, f, max_failure_prob=0.25):
	maxi = L
	for i in range(M,L):
		#sigmas2 = po_rw(L,i,M,f) - mean(i,M)*mean(i,M)
		
		sigmas2 = v(L,i,M,f)
		## a set of size 2^i would "shatter" with probability at least 1-failure_prob
		## using M constraints of density f
		#cantelli
		failure_prob = sigmas2 / (sigmas2 + mean(i,M)*mean(i,M))
		
		# sigmas2N = (po_rw(L,i,M,f)/(mean(i,M)))/(mean(i,M)) -1.0
		# failure_probN = sigmas2N /(sigmas2N +1.0)
		#print failure_prob ,",",failure_prob
		
		if failure_prob<=max_failure_prob:
			if i<maxi:
				maxi=i
			
	return maxi
	
def f_star(L,M, c=2,step=0.01, max_failure_prob=0.25):
	i = M + c
	#if M<c:
	#	return 0.5
	for f in range(1,int(math.ceil(0.5/step))+1):
		sigmas2 = v(L,i,M,step*f)
		#print sigmas2
		#cantelli
		failure_prob = sigmas2 / (sigmas2 + mean(i,M)*mean(i,M))
		#tm = third_moment_rw(L,i,M,f)
		#print "tm=",tm
		#new_fail_prob = 1.0-(math.sqrt(mean(i,M)/tm))**3
		#print "old failure prob", failure_prob, " new failure prob", new_fail_prob
		
		if failure_prob<=max_failure_prob:
			##print "variance ", sigmas2
			return step*f
	return 0.5
	
start = time.time()

__version__ = '0.1'

parser = argparse.ArgumentParser(description='Counting contingency tables')

parser.add_argument('-v', '--version',      action='version', version='%(prog)s ' + __version__)

parser.add_argument('-n', '--n', type=int, help="nXn adjacency table", default=20)

parser.add_argument('-x', '--x', type=int, help="number of xors to add", default=0)

parser.add_argument('-L', '--L', type=int, help="number of LONG xors (full length) to add", default=0)

parser.add_argument('-df', '--df', action="store_true", help="darwin finches", default=False)
parser.add_argument('-icons', '--icons', action="store_true", help="icons dataset", default=False)
parser.add_argument('-purum', '--purum', action="store_true", help="purum dataset", default=False)
parser.add_argument('-frogsM', '--frogsM', action="store_true", help="purum dataset", default=False)
parser.add_argument('-iqd', '--iqd', action="store_true", help="industrial quality control dataset", default=False)

parser.add_argument('-blocked', '--blocked', type=int, help="Use NxN blocked matrix (as in NIPS-13 paper)", default=0)

parser.add_argument('-osmt', '--osmt', action="store_true", help="Write SMTLIB file and exit", default=False)

parser.add_argument('-bv', '--bv', action="store_true", help="Use bitvector encoding", default=True)

parser.add_argument('-option2', '--option2', action="store_true", help="option2 (random non overlapping clauses)", default=False)

parser.add_argument('-f', '--f', type=float, help="density", default=0.5)

parser.add_argument('-seed', '--seed', type=int, help="random seed", default=9189181171)

args = parser.parse_args()

n = args.n
#r=args.r

if args.x>=0:
	xor_num = args.x
else:
	xor_num = 3

random.seed(args.seed)	
#np.random.seed(seed=args.seed)

s = Solver()
#s = SimpleSolver()

nr = n
nc = n

## default marginals
row_sums = [3]*nr
col_sums = [3]*nc
BooleanContigencyTable = False

if args.df:
	# nr = 13
	# nc = 17
	# row_sums = [14, 13,14,10,12,2,10,1,10,11,6, 2, 17]
	col_sums_orig = [4,	4,	11,	10,	10,	8,	9,	10,	8,	9,	3,	10,	4,	7,	9,	3,	3]
	
	## darwin finches data
	nr = 12
	nc = 17
	row_sums = [14, 13,14,10,12,2,10,1,10,11,6, 2]
	col_sums = [i-1 for i in col_sums_orig]

	BooleanContigencyTable = True
	forcedZeros= set()
	
if args.icons:
	nr = 9
	nc = 6
	row_sums = [15,18,16,20,18,11,9,8,18]
	col_sums = [32,24,30,24,14,9]
	forcedZeros = set([(2,0), (5,0), (8,0), (1,1), (4,1), (7,1), (0,2), (3,2), (6,2), (6,3), (7,3), (8,3), (0,4), (1,4), (2,4), (3,5), (4,5), (5,5)]) 

if args.blocked > 0:	
	nr = args.blocked
	nc = args.blocked
	row_sums = [args.blocked-1]*args.blocked
	col_sums = [args.blocked-1]*args.blocked
	row_sums[0]=1
	col_sums[0]=1
	
	BooleanContigencyTable = True
	forcedZeros= set()
	
if args.purum:
	nr = 5
	nc = 5
	row_sums = [28,23,23,19,35]
	col_sums = [21,27,25,26,29]
	forcedZeros = set([(0,0), (0,3), (1,1), (2,0), (2,2), (3,1), (3,2), (3,3)])

if args.iqd:
	nr = 7
	nc = 4
	row_sums = [7,3,4,5,5,4,3]
	col_sums = [12,7,8,4]
	forcedZeros = set([(1,3), (2,2), (3,1),(4,1),(4,2), (5,2),(5,3),(6,1),(6,3)])
	
	
if args.frogsM:
	nr = 9
	nc = 9
	row_sums = [50,51,67,70,81,84,100,107,110]
	col_sums = [110,109,93,90,79,76,60,53,50]
	forcedZeros = set([(0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)])



	
print len(row_sums), 'row sums:', row_sums
print len(col_sums), 'col sums:', col_sums


## max entries for each entry in the contigency table
tot_entries = int(sum(row_sums))
if BooleanContigencyTable:
	tot_entries = 1

nbits = int(math.ceil(math.log(max(max(row_sums),max(col_sums))+1,2.0)))
print nbits
	
bitsPerEntry=nbits
if BooleanContigencyTable:
	bitsPerEntry = 1


	
## generate variables
Ab=[]
A=[]
ABV=[]	
bits_per_entry=[]	
nbr_variables = 0

for i in range(nr):
	a=[]
	ab=[]
	abv=[]
	be=[]
	for ip in range(nc):
		
		# var = Int('A_%d_%d' %(i,ip))
		# s.add(var>=0)
		# if BooleanContigencyTable:
			# s.add(var<=1)
		# a.append(var)
		
		if args.bv:
			bitsPerEntry = int(math.ceil(math.log(min(row_sums[i],col_sums[ip])+1,2.0)))
			if BooleanContigencyTable:
				bitsPerEntry = 1
			av = BitVec('ABV_%d_%d'%(i,ip), bitsPerEntry)
			abv.append(av)
			be.append(bitsPerEntry)
			nbr_variables = nbr_variables +bitsPerEntry
			
			if ((i,ip) in forcedZeros):
				s.add(av==0)
				
			#if BooleanContigencyTable:
			#	s.add(av<=1)
			
		else:
			var = Bool('A_%d_%d' %(i,ip))		
			varint = Int('Ai_%d_%d' %(i,ip))
					
			s.add(varint<=1, varint>=0, If(var,varint==1,varint==0))
			a.append(varint)
			ab.append(var)
				
	A.append(a)
	Ab.append(ab)
	ABV.append(abv)
	bits_per_entry.append(be)
	
print bits_per_entry
# for qq in range(nc):
	# for ww in range(nc):
		# for ss in range(nc):
			# if (qq!=ww and qq!=ss and ww!=ss):
				# s.add(Implies(And(Ab[nr-1][qq],Ab[nr-1][ww]),Not(Ab[nr-1][ss])))

# for qq in range(nc):
	# for ss in range(nc):
		# if (qq!=ss):
			# s.add(Implies(Ab[7][qq],Not(Ab[7][ss])))

				
## add row marginals constraints
for i in range(nr):
	row_marg_bv = 0
	row_marg = 0
	for ip in range(nc):
		#row_marg = row_marg + If(A[i][ip], 1,0)
		if args.bv:
			row_marg_bv = row_marg_bv + ZeroExt(nbits+2-bits_per_entry[i][ip],ABV[i][ip])			# 4 bits
		else:
			row_marg = row_marg + A[i][ip]
		#A = np.random.rand(d,k)*1.0
	if args.bv:
		s.add(row_marg_bv==row_sums[i])
	else:
		s.add(row_marg==row_sums[i])

## add column marginals
for ip in range(nc):
	col_marg = 0
	col_marg_bv = 0
	for i in range(nr):
		#col_marg = col_marg + If(A[i][ip], 1, 0)
		if args.bv:
			col_marg_bv = col_marg_bv + ZeroExt(nbits+2-bits_per_entry[i][ip],ABV[i][ip])			# 4 bits
		else:
			col_marg = col_marg + A[i][ip]

	#A = np.random.rand(d,k)*1.0
	if args.bv:
		s.add(col_marg_bv==col_sums[ip])
	else:
		s.add(col_marg==col_sums[ip])
	

# countpos=0	
# counterConst=0

## get minimum density required as ICML-14 paper
L= n*n
#print f_star(L,xor_num, c=2,step=0.01, max_failure_prob=0.25)

f = args.f	

if args.x>0:
	## generate random xors
	for k in range(xor_num):
		xorscope= False
		varsScope=[]
		if randint(0,1)<=0:
			xorscope=True
		for i in range(nr):
			for ip in range(nc):
				for bitnum in range(bits_per_entry[i][ip]):
					if random.random()<f:
						if args.bv:
							xorscope=xorscope ^ Extract(bitnum, bitnum, ABV[i][ip])
							#xorscope=xorscope ^ ABV[i][ip]
						else:
							xorscope=Xor(xorscope,Ab[i][ip])
							varsScope.append(Ab[i][ip])
				# for z in range(1):
					# if random.random()<f:
						# var = Bool('A_%d_%d==%d' %(i,ip,z))
						# s.add(Implies(A[i][ip]==z,var))
						# s.add(Implies(var,A[i][ip]==z))
						# #s.add(var)
						# xorscope=Xor(xorscope,var)
		if args.option2:
			xor_length = len(varsScope)
			## add 2^{} random, non overlapping clauses
			from itertools import chain, combinations
			varsScopeStr = [str(v) for v in varsScope]
			#print varsScope
			#powerset = [set(q) for q in chain.from_iterable(combinations(varsScope, r) for r in range(len(varsScope)+1))]
			powerset = list(chain.from_iterable(combinations(varsScopeStr, r) for r in range(len(varsScope)+1)))				
			#print powerset
			selectedC = random.sample(powerset, 2**(xor_length-1))
			#print selectedC
			for c in selectedC:
				#sc = str(list(c))
				orscope= False
				#print list(c)
				for v in varsScope:
					#print v
					if str(v) in list(c):
						orscope=Or(orscope,v)
					else:
						orscope=Or(orscope,Not(v))
				#print orscope
				s.add(orscope)
			##for z in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
		else:
			if args.bv:
				s.add(xorscope==1)	
			else:
				s.add(xorscope)

	## generate L random LONG xors
	for k in range(args.L):
		xorscope= False
		if randint(0,1)<=0:
			xorscope=True	
		for i in range(nr):
			for ip in range(nc):
				if random.random()<0.5:
					if args.bv:
						xorscope=xorscope ^ ABV[i][ip]
					else:
						xorscope=Xor(xorscope,Ab[i][ip])
						varsScope.append(Ab[i][ip])
				# for z in range(1):
					# if random.random()<f:
						# var = Bool('A_%d_%d==%d' %(i,ip,z))
						# s.add(Implies(A[i][ip]==z,var))
						# s.add(Implies(var,A[i][ip]==z))
						# #s.add(var)
						# xorscope=Xor(xorscope,var)
		
		## add the XOR
		if args.bv:
			s.add(xorscope==1)	
		else:
			s.add(xorscope)
		
	#print s.help()
	
	file = open("ct.%d.%f.bv%d.%d.smt2"%(xor_num,f,args.bv,args.seed), "w")
	file.write(convertor(s, logic="QF_LIA", name="darwin.xor%d.density%f.bv%d.seed%d"%(xor_num,f,args.bv,args.seed)))
	file.close()
	if args.osmt:
		quit()
	#print convertor(s, logic="QF_LIA")
	
	startsolv = time.time()
	print "time taken to build model", startsolv-start 
	result =  s.check()
	print result
	
	end = time.time()
	print "time taken to solve", end-startsolv 
	if str(result)=='sat':
		m =  s.model()
		print "A="
		for i in range(nr):
			for ip in range(nc):
				if args.bv:
					print str(m[ABV[i][ip]]),",",
				else:	
					print str(m[A[i][ip]]),",",
				# if str(m[A[i][ip]])=="True":
					# print "1","|",
				# else:
					# print "0","|",
			print ""
		print s.statistics()
		#print "violated constraints: ",violatedc
	
	

	
	print "number of variables=", nbr_variables
	print "f=",f, ".Required f*=", f_star(nbr_variables,xor_num, c=2,step=0.01, max_failure_prob=0.45)
	
	if str(result)=='sat':
		print "lower bound, ", xor_num
	else:
		print "upper bound, ", upper_bound_star(nbr_variables,xor_num, f, max_failure_prob=0.45)
	
	
else:
	import math
	xor_num = 0
	scale =-math.log(2.0)
	xlen = 1.0
	print s.check()
	while str(s.check())=='sat':
		xorscope= False
		scale = scale - math.log(2**(xlen)-1)+math.log(2**(xlen))
		xlen = 0
		#if randint(0,1)<=0:
		#	xorscope=True
		for i in range(nr):
			for ip in range(nc):
				if random.random()<f:
					#xorscope=Xor(xorscope,Ab[i][ip])
					xlen = xlen +1
					if random.random()<0.5:
						xorscope=Or(xorscope,Ab[i][ip])
					else:
						xorscope=Or(xorscope,Not(Ab[i][ip]))
		print xorscope
			# for z in range(1):
					# if random.random()<f:
						# var = Bool('A_%d_%d==%d' %(i,ip,z))
						# s.add(Implies(A[i][ip]==z,var))
						# s.add(Implies(var,A[i][ip]==z))
						# #s.add(var)
						# xorscope=Xor(xorscope,var)

		s.add(xorscope)
		xor_num= xor_num +1
		print xor_num,",",scale/math.log(2.0)
	print scale/math.log(2.0)