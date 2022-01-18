#!/usr/bin/env python

import os

import grok

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "./log2"))
hparams = parser.parse_args()
hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)


print(hparams)
print(grok.training.train(hparams))
