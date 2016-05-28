from music21 import *

#Filter for every Bach chorale that is in 4/4:
catalog = stream.Opus()
for workName in corpus.getBachChorales():
  work = converter.parse(workName)
  firstTS = work.flat.getTimeSignatures()[0]
  if firstTS.ratioString == '4/4':
    catalog.append(work)

#Count usages
allbeats = list();
for p in catalog.flat.notes:
   for counter in range(len(p.pitches)):
        allbeats.append(p.beat)

#Create the histogram
import matplotlib.pyplot as plt
import numpy as np
plt.hist(allbeats, bins=32, range=(1,5))
#plt.title("Metrical Position Usage")
plt.xlabel("Metrical Position ('crotchet', or '1/4 note')", fontsize=20)
plt.ylabel("Frequency of Usage", fontsize=20)
plt.xlim(1,5)
plt.xticks(np.arange(1, 5, 1.0))
plt.savefig('Chorale position usage.png', facecolor='w', edgecolor='w', format='png')
plt.show()