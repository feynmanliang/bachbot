#!/usr/bin/env python
"""
Usage: jug <command> scripts/quiz.py
"""

import os
import os.path
import glob
import subprocess
import json
from collections import defaultdict
from jug import Task

from music21 import *
from constants import *

import numpy as np
np.random.seed(42)

num_questions = 3 * 4 * 2
HARM_OUT_DIR = SCRATCH_DIR + '/harm_out'
QUESTIONS_DIR = SCRATCH_DIR + '/quiz'

FNULL = open(os.devnull, 'w')

def fix_repeating_semiquavers(s):
    "Joins together repeated semiquavers at the same pitch."
    new_score = copy.deepcopy(s)
    for m in new_score.parts[0].getElementsByClass('Measure'):
        for start in np.arange(0.0, 4.0, 0.5):
            chords = m.notesAndRests.getElementsByOffset(start, start+0.5, includeEndBoundary=False)
            if len(chords) > 1:
                prev_el = chords[0]
                el = chords[1]
                prev_el.removeRedundantPitches(inPlace=True)
                el.removeRedundantPitches(inPlace=True)

                for prev_n, n in zip(sorted(prev_el._notes), sorted(el._notes)):
                    if prev_n.pitch == n.pitch:
                        if not prev_n.tie or prev_n.tie.type == 'stop':
                            prev_n.tie = tie.Tie('start')
                            n.tie = tie.Tie('stop')
    return new_score

def xml_to_mp3(xml_path, out_dir):
    """
    Converts a file at `xml_path` ending in '.xml' to a '.mp3' in `out_dir`.
    """
    if not out_dir:
        out_dir = os.path.dirname(xml_path)
    out_path = os.path.join(out_dir, os.path.basename(xml_path[:-3]+'mp3'))
    subprocess.call([
        'mscore',
        '-o', out_path,
        xml_path,
    ], stdout=FNULL, stderr=subprocess.STDOUT)
    return out_path

def make_question(orig_fp, gen_fp, out_dir):
    print 'Now processing: {}'.format(gen_fp)
    orig_mp3_fname = os.path.basename(xml_to_mp3(orig_fp, out_dir))
    gen_mp3_fname = os.path.basename(xml_to_mp3(gen_fp, out_dir))
    return {
            'original': orig_mp3_fname,
            'generated': gen_mp3_fname
            }

if not os.path.exists(QUESTIONS_DIR):
    os.mkdir(QUESTIONS_DIR)

# Prepares all harmonization example question pairs
orig_fps = glob.glob(HARM_OUT_DIR + '/*-nomask-fermatas.xml')
np.random.shuffle(orig_fps)
single_part_masks = ['Soprano', 'Alto', 'Tenor', 'Bass']
multi_part_masks = ['Alto-Tenor', 'Alto-Tenor-Bass']

question_groups = defaultdict(list)
for i, orig_fp in enumerate(orig_fps[:num_questions]):
    root_fp = (os.path.dirname(orig_fp) + '/' + '-'.join(os.path.basename(orig_fp).split('-')[:2]))
    if i % (1+len(multi_part_masks)) == len(multi_part_masks):
        masked_parts = single_part_masks[int(i / (1+len(multi_part_masks))) % len(single_part_masks)]
    else:
        masked_parts = multi_part_masks[i % (1+len(multi_part_masks))]
    gen_fp = root_fp + '-mask-{}-fermatas.xml'.format(masked_parts)
    question_groups[masked_parts].append(Task(make_question, orig_fp, gen_fp, QUESTIONS_DIR))

# Prepare all compare against generative samples
for orig_fp, gen_fp in zip(orig_fps[num_questions:], glob.glob(SCRATCH_DIR + '/out/out-*.xml')):
    # fix repeated semiquavers
    gen_fixed_fp = QUESTIONS_DIR + '/' + os.path.basename(gen_fp)[:-4] + '-fixed.xml'
    fix_repeating_semiquavers(converter.parseFile(gen_fp)).write('xml', gen_fixed_fp)

    question_groups['AllParts'].append(Task(make_question, orig_fp, gen_fixed_fp, QUESTIONS_DIR))


