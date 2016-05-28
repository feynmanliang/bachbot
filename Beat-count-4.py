#Open music21:
from music21 import *

#Tell music21 to access a score. Either:
#... from its own corpus eg:
#s = corpus.parse("bach/bwv846")
#... or from elseshere on your computer. eg:
s = converter.parse('~/Desktop/Score-name.xml')

#Count usage of all metrical positions (one count for each pitch in a chord):
allbeats = list();
for p in s.flat.notes:
   for counter in range(len(p.pitches)):
        allbeats.append(p.beat)
#print(allbeats)

#Create the histogram
import matplotlib.pyplot as plt
import numpy as np
#'bins' for the number of positions (e.g. 8, 16, or 32 for 4/4 metres).
plt.hist(allbeats, bins=32, range=(1,5))
#plt.title("Metrical Position Usage")
plt.xlabel("Metrical Position in 'crotchets' ('1/4 note')", fontsize=20)
plt.ylabel("Frequency of Usage", fontsize=20)
plt.xlim(1,5)
plt.xticks(np.arange(1, 5, 1.0))

plt.savefig('Score-name.png', facecolor='w', edgecolor='w', format='png')
plt.show()