#!/usr/bin/env python

import os
import grok
from grok import training_ptb

parser = training_ptb.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "./logs/"))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)

print(hparams)
print(training_ptb.train(hparams))
print("Finished")
