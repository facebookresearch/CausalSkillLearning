# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .headers import *
from . import MIME_DataLoader

opts = flags.FLAGS

def main(_):

	dataset = MIME_DataLoader.MIME_Dataset(opts)
	print("Created DataLoader.")

	embed()

if __name__ == '__main__':
	app.run(main)