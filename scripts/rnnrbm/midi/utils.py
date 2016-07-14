# Author: Nicolas Boulanger-Lewandowski
# University of Montreal (2013)
# RNN-RBM deep learning tutorial
#
# Implements midiread and midiwrite functions to read/write MIDI files to/from piano-rolls


from MidiOutFile import MidiOutFile
from MidiInFile import MidiInFile
from MidiOutStream import MidiOutStream

import numpy


class midiread(MidiOutStream):
  def __init__(self, filename, r=(21, 109), dt=0.2):
    self.notes = []
    self._tempo = 500000
    self.beat = 0
    self.time = 0.0

    midi_in = MidiInFile(self, filename)
    midi_in.read()
    self.notes = [n for n in self.notes if n[2] is not None]  # purge incomplete notes

    length = int(numpy.ceil(max(zip(*self.notes)[2]) / dt))  # create piano-roll
    self.piano_roll = numpy.zeros((length, r[1]-r[0]))
    for n in self.notes:
      self.piano_roll[int(numpy.ceil(n[1]/dt)) : int(numpy.ceil(n[2]/dt)), n[0]-r[0]] = 1

  def abs_time_in_seconds(self):
    return self.time + self._tempo * (self.abs_time() - self.beat) * 1e-6 / self.div

  def tempo(self, value):
    self.time = self.abs_time_in_seconds()
    self.beat = self.abs_time()
    self._tempo = value
  
  def header(self, format=0, nTracks=1, division=96):
    self.div = division

  def note_on(self, channel=0, note=0x40, velocity=0x40):
    self.notes.append([note, self.abs_time_in_seconds(), None])

  def note_off(self, channel=0, note=0x40, velocity=0x40):
    i = len(self.notes) - 1
    while i >= 0 and self.notes[i][0] != note:
      i -= 1
    if i >= 0 and self.notes[i][2] is None:
      self.notes[i][2] = self.abs_time_in_seconds()

  def sysex_event(*args):
    pass

  def device_name(*args):
    pass


def midiwrite(filename, piano_roll, r=(21, 109), dt=0.2, patch=0):
  midi = MidiOutFile(filename)
  midi.header(division=100)
  midi.start_of_track() 
  midi.patch_change(channel=0, patch=patch)
  t = 0
  samples = [i.nonzero()[0] + r[0] for i in piano_roll]

  for i in xrange(len(samples)):
    for f in samples[i]:
      if i==0 or f not in samples[i-1]:
        midi.update_time(t)
        midi.note_on(channel=0, note=f, velocity=90)
        t = 0
    
    t += int(dt*200)

    for f in samples[i]:
      if i==len(samples)-1 or f not in samples[i+1]:
        midi.update_time(t)
        midi.note_off(channel=0, note=f, velocity=0)
        t = 0
      
  midi.update_time(0)
  midi.end_of_track()
  midi.eof()


