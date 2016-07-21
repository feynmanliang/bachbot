from music21 import *

#Open a score:
#s = corpus.parse("bach/bwv846")
s = converter.parse('~/Desktop/1.xml')

p = graph.PlotHistogramPitchClass(s)
p.id
'histogram-pitchClass'
p.process()