# -*- coding: utf-8 -*-

from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
from CuAsm.CuInsFeeder import CuInsFeeder

sassname = 'TestData/Mandelbrot.sm_75.sass'

# initialize a feeder with sass
feeder = CuInsFeeder(sassname)

# initialize an empty repos
repos = CuInsAssemblerRepos()

# Update the repos with instructions from feeder
repos.update(feeder)

# reset the feeder back to start
feeder.restart()

# verify the repos
# actually the codes is already verifed during repos construction
repos.verify(feeder)