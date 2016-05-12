#!/usr/bin/env python

import music21
m = music21.converter.parse("ana-music/corpus/mozart/sonata07-1.krn")
m.show("musicxml")
