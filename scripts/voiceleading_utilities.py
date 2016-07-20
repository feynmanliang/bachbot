import random

_VERYLARGENUMBER = 1000000				# effectively infinity
_MODULUS = 12							# size of the octave

_HALFMODULUS = int(0.5 + _MODULUS/2.0)

"""

voiceleading_utilities version 1.0, (c) 2015 by Dmitri Tymoczko

Voiceleading_utilities is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License 
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. Voiceleading_utilities 
is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.   You should have received a copy of the GNU Lesser 
General Public License along with Voiceleading_utilities.  If not, see <http://www.gnu.org/licenses/>. 

A set of routines that identify the minimal voice leadings between sets of pitches and pitch classes.

1. bijective_vl finds the best bijective voice leading between pitch-class sets assuming a fixed number of voices
	- use this if you want to control the number of voices exactly, 
		e.g., a 3-voice voice leading from [C, E, G] to [F, A, C], or
		a 4-voice voice leading from [G, B, D, F] to [C, C, E, G]
	- this routine will also rank all the voice-leadings by size, so that, e.g. you can use the second-most efficient voice leading if you want
	NB: this routine passes back pairs of the form [startPC, path]
		
2. voicelead takes an input set of pitches and a target set of PCs, and outputs a set of pitches;
	- this is useful if you are generating music, and have a specific C-major chord in register; it will tell you where each voice should go
	there is an option here to randomly choose one of the N most efficient voice leadings, so you are not always restricted to the most efficient ones
	
3. nonbijective_vl allows notes to be doubled; sometimes this produces a more efficient voice leading than a bijective voice leading
	- for this reason, you cannot always control the number of voices
	NB: this routine passes back pairs of PCs, from which you may need to calculate paths

(For details on the nonbijective_vl algorithm, see Tymozko, D., "The Geometry of Musical Chords", Science, 2006.)

Sometimes, you want something in between, e.g. the best 4-voice voice leading between triads or from a 4-voice seventh to a triad; in this case, 
you need to iterate bijective_vl over all possible doublings of the chords.  This can be time consuming.

TODO: allow different choices of metric

"""

"""==============================================================================================================================

bijective_vl expects two SORTED equal-length sets of integers representing PCs (in any modulus).
the sort parameter sorts the possible bijective VLs by size; by default it is set to False.  Set it to true only if you want to choose from
	among the n most efficient VLs"""

def bijective_vl(firstPCs, secondPCs, sort = False):
	if len(firstPCs) != len(secondPCs):
		return False
	bijective_vl.fullList = []										# collects all the bijective VLs along with their size
	currentBest = []												# currentBest records the best VL we have found so far
	currentBestSize = _VERYLARGENUMBER								# currentBestSize is the size of the current best VL (starts at infinity)
	for i in range(0, len(firstPCs)):								# iterate through every inversion of the  second PC
		secondPCs = secondPCs[-1:] + secondPCs[:-1]
		newSize = 0	
		newPaths = []
		for i in range(0, len(firstPCs)):
			path = (secondPCs[i] - firstPCs[i]) % _MODULUS			# calculate most efficient path based on the pairs
			if path > _HALFMODULUS: 								# negative numbers for descending paths
				path -= _MODULUS
			newPaths.append([firstPCs[i], path])
			newSize += abs(path)
		bijective_vl.fullList.append([newPaths, newSize])		
		if newSize < currentBestSize:								# record the current best size
			currentBestSize = newSize
			currentBest = newPaths
	bijective_vl.size = currentBestSize
	if sort:
		bijective_vl.fullList = sorted(bijective_vl.fullList, key = lambda x: x[1])
	return currentBest

"""==============================================================================================================================

voicelead expects a source list of PITCHES and a target list of PCs, both should be the same length; it outputs one of the topN most efficient voice leadings
from the source pitches to the target PCs.  

if topN is 1, it gives you the most efficient voice leading"""

def voicelead(inPitches, targetPCs, topN = 1):
	inPCs = sorted([p % _MODULUS for p in inPitches])							# convert input pitches to PCs and sort them
	targetPCs = sorted(targetPCs)
	paths = bijective_vl(inPCs, targetPCs, topN != 1)							# find the possible bijective VLs
	if topN != 1:																# randomly select on of the N most efficient possibilities
		myRange = min(len(bijective_vl.fullList), topN)
		paths = bijective_vl.fullList[random.randrange(0, myRange)][0]
	output = []
	tempPaths = paths[:]														# copy the list of paths
	for inPitch in inPitches:
		for path in tempPaths:													# when we find a path remove it from our list (so we don't duplicate paths)
			if (inPitch % _MODULUS) == path[0]:
				output.append(inPitch + path[1])
				tempPaths.remove(path)
				break
	return output

"""==============================================================================================================================

nonbijective_vl expects a source list of PCs or pitches and a target list of PCs or pitches, of any lengths; it outputs the most efficient voice leading from 
source to target.  Voices can be arbitrarily doubled.  

To see why this is interesting, compare bijective_vl([0, 4, 7, 11], [4, 8, 11, 3]) to nonbijective_vl([0, 4, 7, 11], [4, 8, 11, 3])

for PCs, nonbijective_vl iterates over every inversion of the target chord; for each inversion it builds a matrix showing the most efficient voice leading
such that the first note of source goes to the first note of target (see Tymoczko "The Geometry of Musical Chords" for details)

TODO: choose the smaller of source and target to iterate over??

"""

def nonbijective_vl(source, target, pcs = True):
	curVL = []
	curSize = _VERYLARGENUMBER
	if pcs: 
		source = [x % _MODULUS for x in source]
		target = [x % _MODULUS for x in target]
	source = sorted(list(set(source)))
	target = sorted(list(set(target)))
	if pcs:
		for i in range(len(target)):								# for PCs, iterate over every inversion of the target
			tempTarget = target[i:] + target[:i]					
			newSize = build_matrix(source, tempTarget)				# generate the matrix for this pairing
			if newSize < curSize:									# save it if it is the most efficient we've found
				curSize = newSize
				curVL = find_matrix_vl()
		curVL = curVL[:-1]
	else:
		curSize = build_matrix(source, tempTarget)					# no need to iterate for pitches
		curVL = find_matrix_vl()
	return curSize, curVL

def build_matrix(source, target, pcs = True):						# requires sorted source and target chords
	global theMatrix
	global outputMatrix
	global globalSource
	global globalTarget
	if pcs: 
		source = source + [source[0]]
		target = target + [target[0]]
		distanceFunction = lambda x, y: min((x - y) % _MODULUS, (y - x) % _MODULUS)		# add **2 for Euclidean distance
	else:
		distanceFunction = lambda x, y: abs(x - y)
	globalSource = source
	globalTarget = target
	theMatrix = []
	for targetItem in target:
		theMatrix.append([])
		for sourceItem in source:
			theMatrix[-1].append(distanceFunction(targetItem, sourceItem))
	outputMatrix = [x[:] for x in theMatrix]
	for i in range(1, len(outputMatrix[0])):
		outputMatrix[0][i] += outputMatrix[0][i-1]
	for i in range(1, len(outputMatrix)):
		outputMatrix[i][0] += outputMatrix[i-1][0]
	for i in range(1, len(outputMatrix)):
		for j in range(1, len(outputMatrix[i])):
			outputMatrix[i][j] += min([outputMatrix[i][j-1], outputMatrix[i-1][j], outputMatrix[i-1][j-1]])
	return outputMatrix[i][j] - theMatrix[i][j]
			
def find_matrix_vl():							# identifies the voice leading for each matrix
	theVL = []
	i = len(outputMatrix) - 1
	j = len(outputMatrix[i-1]) - 1
	theVL.append([globalSource[j], globalTarget[i]])
	while (i > 0 or j > 0): 
		newi = i
		newj = j
		myMin = _VERYLARGENUMBER
		if i > 0 and j > 0:
			newi = i - 1
			newj = j - 1
			myMin = outputMatrix[i-1][j-1]
			if outputMatrix[i-1][j] < myMin:
				myMin = outputMatrix[i-1][j]
				newj = j
			if outputMatrix[i][j - 1] < myMin:
				myMin = outputMatrix[i][j-1]
				newi = i
			i = newi
			j = newj
		elif i > 0:
			i = i - 1
		elif j > 0:
			j = j - 1
		theVL.append([globalSource[j], globalTarget[i]])
	return theVL[::-1]

"""==============================================================================================================================

A simple routine to put voice leadings in 'normal form.'  Essentially, we just apply the standard "left-packing" algorithm to the first element
in a list of [startPC, path] pairs.

"""

def vl_normal_form(inList):														# list of [PC, path] pairs
	myList = sorted([[k[0] % _MODULUS] + k[1:] for k in inList])
	currentBest = [[(k[0] - myList[0][0]) % _MODULUS] + k[1:] for k in myList]
	vl_normal_form.transposition = myList[0][0] * -1
	for i in range(1, len(myList)):
		newChallenger = myList[-i:] + myList[:-i]
		transp = newChallenger[0][0] * -1
		newChallenger = sorted([[(k[0] - newChallenger[0][0]) % _MODULUS] + k[1:] for k in newChallenger])
		for j in reversed(range(len(myList))):
			if newChallenger[j][0] < currentBest[j][0]:
				currentBest = newChallenger
				vl_normal_form.transposition = transp
			else:
				if newChallenger[j][0] > currentBest[j][0]:
					break
	return currentBest