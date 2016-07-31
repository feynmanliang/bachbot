import os.path
import glob
import subprocess
import json

import numpy as np

from constants import *
from jug import Task

num_questions = 9
HARM_OUT_DIR = SCRATCH_DIR + '/harm_out'
HARM_QUESTIONS_DIR = SCRATCH_DIR + '/harm_questions'

if not os.path.exists(HARM_QUESTIONS_DIR):
    os.mkdir(HARM_QUESTIONS_DIR)

def xml_to_mp3(xml_path, out_dir):
    "Converts a file at `xml_path` ending in '.xml' to a '.mp3' in `out_dir`."
    if not out_dir:
        out_dir = os.path.dirname(xml_path)
    out_path = os.path.join(out_dir, os.path.basename(xml_path[:-3]+'mp3'))
    subprocess.call([
        'mscore',
        '-o', out_path,
        xml_path,
    ])
    return out_path

def make_question(orig_fp, gen_fp, out_dir):
    print 'Now processing: {} with mask {}'.format(orig_fp, masked_parts)
    orig_mp3_fname = os.path.basename(xml_to_mp3(orig_fp, out_dir))
    gen_mp3_fname = os.path.basename(xml_to_mp3(gen_fp, out_dir))
    return {
	    'original': orig_mp3_fname,
	    'generated': gen_mp3_fname
	    }

# Prepares all harmonization example question pairs
orig_fps = glob.glob(HARM_OUT_DIR + '/*-nomask-fermatas.xml')
single_part_masks = ['Soprano', 'Alto', 'Tenor', 'Bass']
multi_part_masks = ['Alto-Tenor', 'Alto-Tenor-Bass']

question_pairs = []
for i, orig_fp in enumerate(np.random.choice(orig_fps, size=num_questions, replace=False)):
    root_fp = (os.path.dirname(orig_fp) + '/' + '-'.join(os.path.basename(orig_fp).split('-')[:2]))
    if i % (1+len(multi_part_masks)) == len(multi_part_masks):
        masked_parts = single_part_masks[int(i / (1+len(multi_part_masks))) % len(single_part_masks)]
    else:
        masked_parts = multi_part_masks[i % (1+len(multi_part_masks))]
    gen_fp = root_fp + '-mask-{}-fermatas.xml'.format(masked_parts)
    question_pairs.append(Task(make_question, orig_fp, gen_fp, HARM_QUESTIONS_DIR))
