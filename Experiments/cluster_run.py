"""
Wrapper script for launching a job on the fair cluster.
Sample usage:
      python cluster_run.py --name=trial --setup='/path/to/setup.sh' --cmd='job_command'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from absl import app
from absl import flags
import os
import sys
import random
import string
import datetime
import re

opts = flags.FLAGS

flags.DEFINE_integer('nodes', 1, 'Number of nodes per task')
flags.DEFINE_integer('ntp', 1, 'Number of tasks per node')
flags.DEFINE_integer('ncpus', 40, 'Number of cpu cores per task')
flags.DEFINE_integer('ngpus', 1, 'Number of gpus per task')

flags.DEFINE_string('name', '', 'Job name')
flags.DEFINE_enum('partition', 'learnfair', ['dev', 'priority','uninterrupted','learnfair'], 'Cluster partition')
flags.DEFINE_string('comment', 'for ICML deadline in 2020.', 'Comment')
flags.DEFINE_string('time', '72:00:00', 'Time for which the job should run')

flags.DEFINE_string('setup', '/private/home/tanmayshankar/Research/Code/Setup.bash', 'Setup script that will be run before the command')
# flags.DEFINE_string('workdir', os.getcwd(), 'Job command')
flags.DEFINE_string('workdir', '/private/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments', 'Jod command')
# flags.DEFINE_string('workdir', '/private/home/tanmayshankar/Research/Code/SkillsfromDemonstrations/Experiments/BidirectionalInfoModel/', 'Job command')
flags.DEFINE_string('cmd', 'echo $PWD', 'Directory to run job from')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(_):
    job_folder = '/checkpoint/tanmayshankar/jobs/' + datetime.date.today().strftime('%y_%m_%d')
    mkdir(job_folder)

    if len(opts.name) == 0:
        # read name from command
        opts.name = re.search('--name=\w+', opts.cmd).group(0)[7:]
    print(opts.name)
    slurm_cmd = '#!/bin/bash\n\n'
    slurm_cmd += '#SBATCH --job-name={}\n'.format(opts.name)
    slurm_cmd += '#SBATCH --output={}/{}-%j.out\n'.format(job_folder, opts.name)
    slurm_cmd += '#SBATCH --error={}/{}-%j.err\n'.format(job_folder, opts.name)
    # slurm_cmd += '#SBATCH --exclude=learnfair2038'
    slurm_cmd += '\n'

    slurm_cmd += '#SBATCH --partition={}\n'.format(opts.partition)
    if len(opts.comment) > 0:
        slurm_cmd += '#SBATCH --comment="{}"\n'.format(opts.comment)
    slurm_cmd += '\n'

    slurm_cmd += '#SBATCH --nodes={}\n'.format(opts.nodes)
    slurm_cmd += '#SBATCH --ntasks-per-node={}\n'.format(opts.ntp)
    if opts.ngpus > 0:
        slurm_cmd += '#SBATCH --gres=gpu:{}\n'.format(opts.ngpus)
    slurm_cmd += '#SBATCH --cpus-per-task={}\n'.format(opts.ncpus)
    slurm_cmd += '#SBATCH --time={}\n'.format(opts.time)
    slurm_cmd += '\n'

    slurm_cmd += 'source {}\n'.format(opts.setup)
    slurm_cmd += 'cd {} \n\n'.format(opts.workdir)
    slurm_cmd += '{}\n'.format(opts.cmd)

    job_fname = '{}/{}.sh'.format(job_folder, ''.join(random.choices(string.ascii_letters, k=8)))

    with open(job_fname, 'w') as f:
        f.write(slurm_cmd)

    #print('sbatch {}'.format(job_fname))
    os.system('sbatch {}'.format(job_fname))


if __name__ == '__main__':
    app.run(main)

