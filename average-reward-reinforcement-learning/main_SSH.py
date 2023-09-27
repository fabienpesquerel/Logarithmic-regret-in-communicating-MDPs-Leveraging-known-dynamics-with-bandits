from experiments.fullExperiment import *

import environments.RegisterEnvironments as bW

import learners.Generic.Qlearning as ql
import learners.discreteMDPs.IMEDKTP as IMEDKTP
import learners.discreteMDPs.TSKD as TSKD
import learners.discreteMDPs.UCBKD as UCBKD
import learners.discreteMDPs.UCRL3KTP as UCRL3KTP
import learners.discreteMDPs.PSRLKTP as PSRLKTP

#######################
# Pick an environment
#######################

env = bW.makeWorld(bW.registerWorld('ergo-river-swim-6'))
# env = bW.makeWorld(bW.registerWorld('grid-2-room'))
# env = bW.makeWorld(bW.registerWorld('grid-4-room'))
# env = bW.makeWorld(bW.registerWorld('ergo-river-swim-25'))
# env = bW.makeWorld(bW.registerWorld('nasty'))


nS = env.observation_space.n
nA = env.action_space.n
delta = 0.05

#######################
# Select tested agents
#######################

agents = []

agents.append([IMEDKTP.IMEDKTP, {"env": env}])  # IMED-KD
agents.append([UCBKD.UCBKD, {"env": env}])  # UCB-KD
# agents.append( [UCRL3KTP.UCRL3_lazy, {"nS":nS, "nA":nA, "env":env, "delta":delta}])  # UCRL3
# agents.append( [PSRLKTP.PSRLKTP, {"nS":nS, "nA":nA, "env":env, "delta":delta}])  # PSRL
# agents.append([ql.Qlearning, {"nS": nS, "nA": nA}])  # Q-learning
agents.append([TSKD.TSKD, {"env": env}])  # TS-KD


#######################
# Run a full experiment
#######################
runLargeMulticoreExperiment(env, agents, timeHorizon=20000, nbReplicates=32)
