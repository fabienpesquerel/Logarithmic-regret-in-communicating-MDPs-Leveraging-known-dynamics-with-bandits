import matplotlib.pyplot as plt
from itertools import product
from itertools import chain
import numpy as np
from math import log,sqrt



def dPolicyDistributions(S, A):  # S : states, A : actions
    ''' Distributions of deterministic policies '''
    ns = len(S)
    na = len(A)

    dPolicySet = set(product(set(A), repeat = ns ))
    distributions = []

    for policy in dPolicySet:
        m = np.matrix(np.zeros((ns,na)))
        for s in S:
            m[s, policy[s]] = 1

        distributions = distributions + [ m ]

    return distributions

def randomPolicyDistribution(S, A):  # S : states, A : actions
    ''' Distributions of deterministic policies '''
    ns = len(S)
    na = len(A)

    m = np.matrix(np.zeros((ns,na))) + 1/na

    return m




# def invariant_measure(Pa):
#     nbS = len(Pa)
#     # From https://towardsdatascience.com/markov-chain-analysis-and-simulation-using-python-4507cee0b06e
#     # E.g. with
#     # Pa = np.array([[0.2, 0.7, 0.1],
#     #                [0.9, 0.0, 0.1],
#     #                [0.2, 0.8, 0.0]])
#
#     A = np.append(np.transpose(Pa) - np.identity(nbS), [np.ones(nbS)], axis=0)
#     one = np.zeros(nbS+1)
#     one[-1]=1.
#     print("A",A,"rank:", np.linalg.matrix_rank(A))
#     #print("rank:", np.linalg.matrix_rank(np.append(A,np.transpose([one]),axis=1)))
#     #print(A)
#     #print(b)
#     pi = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(np.transpose(one))) # Rouché–Capelli
#     #print("Invariant measure:", pi, "(Check:",pi.dot(Pa),")")
#
#     print(pi, pi.dot(Pa)) # Checking pi is indeed invariant measure: this sould be the same.
#     #print(pi, Pa.dot(pi)) # Note that these quantities are usually different.
#     return pi

def invariant_measure_cesaro(startingstate,Pa,meanPower=1000,precision=1e-6):
    ns = len(Pa)
    pi = np.zeros(ns)
    pi[startingstate]=1.
    q_t = pi
    q_pow = pi

    i=1.
    while (i<=meanPower and np.max(np.abs(q_t- q_t.dot(Pa))) > precision):
        i=i+1
        alpha = 1./i
        q_pow= q_pow.dot(Pa)
        q_t = (1-alpha)*q_t + alpha*q_pow


    #print(q_t, q_t.dot(Pa), np.max(np.abs(q_t- q_t.dot(Pa)))) # Checking pi is indeed invariant measure: this sould be the same.
    #print(pi, Pa.dot(pi)) # Note that these quantities are usually different.
    return q_t



def getFrequencies(env,policyDistributions,meanPower=1000,precision=1e-6):
    ns, na = policyDistributions[0].shape
    nbPolicies = len(policyDistributions)

    mdpTransitions = np.zeros((ns, na, ns))

    for s in range(ns):
        for a in range(na):
            mdpTransitions[s, a, :] =  env.getTransition(s, a)

    policyFrequencies = np.zeros((ns,nbPolicies, ns, na))
    minFrequencies = np.ones((ns,nbPolicies))#Initialize to highest value
    for p in range(nbPolicies):
        policy = policyDistributions[p]
        Ppolicy = np.zeros((ns,ns))
        for s in range(ns):
            for ss in range(ns):
                # TODO: Carefully check P(s,s') vs P(s',s) !!!
                Ppolicy[ss,s] = np.sum([mdpTransitions[s,a,ss]*policy[s,a] for a in range(na)])
                # v.dot(P) [s] = sum_s1 v(s1) P[s,s1]  when used in invariant_measure_cesaro

        for inistate in range(ns):
            pi = invariant_measure_cesaro(inistate,Ppolicy,meanPower,precision)
            for s in range(ns):
                for a in range(na):
                    policyFrequencies[inistate,p, s, a] = pi[s]*policy[s,a]
                    if (policyFrequencies[inistate,p, s, a]>0):
                        minFrequencies[inistate,p] = min( minFrequencies[inistate,p], policyFrequencies[inistate,p, s, a])

    #
    # policyMatrices = []
    # for distribution in policyDistributions:
    #     matrice = np.matrix(np.zeros((ns*na,ns*na)))
    #     for r in range(ns*na):
    #         s = int(r//na)
    #         a = int(r - s*na)
    #         for c in range(ns*na):
    #             ss = int(c//na)
    #             aa = int(c - ss*na)
    #
    #             matrice[r,c]= distribution[s,a] * mdpTransitions[s,a,ss] * distribution[ss,aa]
    #
    #     policyMatrices = policyMatrices + [matrice]
    #
    #
    # cesaroPolicyMatrices = []
    # for matrice in policyMatrices :
    #     cesaroPolicyMatrices = cesaroPolicyMatrices + [sum([ np.linalg.matrix_power(matrice, power)  for power in range(meanPower)])/meanPower]
    #
    #
    # policyFrequencies = np.zeros((ns, nbPolicies,ns,na))
    # minFrequencies = np.zeros((ns, nbPolicies))
    # for inistate in range(ns):
    #     for p in range(nbPolicies):
    #         minF = np.Inf
    #         distribution = policyDistributions[p]
    #         m = cesaroPolicyMatrices[p]
    #         q = np.matrix(np.zeros((1,ns*na)))
    #         for aa in range(na):
    #             q[0,int(inistate*na+aa)] = distribution[inistate,aa]
    #         for s in range(ns):
    #             for a in range(na):
    #                 d = np.matrix(np.zeros((ns*na,1)))
    #                 d[int(s*na+a),0] = 1
    #                 policyFrequencies[inistate, p, s, a] =  round(np.matmul(np.matmul(q, m),d)[0,0],digitPrecision)
    #                 if policyFrequencies[inistate, p, s, a] > 0:
    #                     minF = min(minF, policyFrequencies[inistate, p, s, a])
    #         minFrequencies[inistate,p] = minF

    return policyFrequencies, minFrequencies

def getFrequencies2(env,policyDistributions,meanPower=1000,precision=1e-6):
    ns, na = policyDistributions[0].shape
    nbPolicies = len(policyDistributions)

    mdpTransitions = np.zeros((ns, na, ns))

    for s in range(ns):
        for a in range(na):
            p = env.getTransition(s, a)
            mdpTransitions[s, a, :] = p
            #for ss in range(ns):
            #    mdpTransitions[s, a, ss] = p[ss]

    policyStateActionFrequencies = np.zeros((ns,nbPolicies, ns, na))
    policyStateFrequencies = np.ones((ns,nbPolicies,ns))
    for p in range(nbPolicies):
        policy = policyDistributions[p]
        Ppolicy = np.zeros((ns,ns))
        for s in range(ns):
            for ss in range(ns):
                Ppolicy[s,ss] = np.sum([mdpTransitions[s,a,ss]*policy[s,a] for a in range(na)])

        for inistate in range(ns):
            pi = invariant_measure_cesaro(inistate,Ppolicy,meanPower,precision)
            for s in range(ns):
                policyStateFrequencies[inistate,p, s]= pi[s]
                for a in range(na):
                    policyStateActionFrequencies[inistate,p, s, a] = pi[s]*policy[s,a]


    return policyStateActionFrequencies, policyStateFrequencies


def getMDPFrequencies(env, policyDistributions, meanPower=1000,digit_precision=7):
    ns, na = policyDistributions[0].shape

    nbPolicies = len(policyDistributions)

    mdpTransitions = np.zeros((ns, na, ns))

    for s in range(ns):
        for a in range(na):
            for l in env.P[s][a]:
                ss = l[1]
                mdpTransitions[s, a, ss] = l[0]


    policyMatrices = []
    for distribution in policyDistributions:
        matrice = np.matrix(np.zeros((ns * na, ns * na)))
        for r in range(ns * na):
            s = int(r // na)
            a = int(r - s * na)
            for c in range(ns * na):
                ss = int(c // na)
                aa = int(c - ss * na)

                matrice[r, c] = distribution[s, a] * mdpTransitions[s, a, ss] * distribution[ss, aa]

        policyMatrices = policyMatrices + [matrice]


    cesaroPolicyMatrices = []
    for matrice in policyMatrices:
        cesaroPolicyMatrices = cesaroPolicyMatrices + [
            sum([np.linalg.matrix_power(matrice, power) for power in range(meanPower)]) / meanPower]


    frequencies = np.zeros((ns, nbPolicies, ns, na))
    minFrequencies = np.zeros((ns, nbPolicies))
    for ss in range(ns):
        for p in range(nbPolicies):
            minF = np.Inf
            distribution = policyDistributions[p]
            m = cesaroPolicyMatrices[p]
            q = np.matrix(np.zeros((1, ns * na)))
            for aa in range(na):
                q[0, int(ss * na + aa)] = distribution[ss, aa]
            for s in range(ns):
                for a in range(na):
                    d = np.matrix(np.zeros((ns * na, 1)))
                    d[int(s * na + a), 0] = 1
                    frequencies[ss, p, s, a] = round(np.matmul(np.matmul(q, m), d)[0, 0], digit_precision)
                    if frequencies[ss, p, s, a] > 0:
                        minF = min(minF, frequencies[ss, p, s, a])
            minFrequencies[ss, p] = minF
    #
    # mdp.policyFrequencies = frequencies
    # mdp.minFrequencies = minFrequencies
    #
    # gains = np.zeros((ns, nbPolicies))
    # for ss in range(ns):
    #     for p in range(nbPolicies):
    #         g = 0
    #         for s in range(ns):
    #             for a in range(ns):
    #                 g = g + frequencies[ss, p, s, a] * mdp.R[s][a].mean()
    #
    #         gains[ss, p] = g
    #
    # mdp.policyGains = gains
    #
    # maxGains = np.zeros(ns)
    # for ss in range(ns):
    #     maxGains[ss] = max(gains[ss])
    #
    # mdp.maxPolicyGains = maxGains
    #
    # gaps = np.zeros((ns, nbPolicies))
    # for ss in range(ns):
    #     for p in range(nbPolicies):
    #         gaps[ss, p] = maxGains[ss] - gains[ss, p]
    #
    # mdp.policyGaps = gaps

    return frequencies, minFrequencies



class TSKTP:
    '''Index Minimum Empirical Divergence for MDPs with Known Transition Probabilities'''

    def __init__(self, env):
        self.mdp = env
        self.nS =  env.observation_space.n
        self.nA = env.action_space.n
        self.states = [s for s in range(self.nS)]
        self.actions = [a for a in range(self.nA)]

        self.policyDistributions = [randomPolicyDistribution(self.states, self.actions)] + dPolicyDistributions(self.states,
                                                                                                     self.actions)
        self.nP = len(self.policyDistributions)


        self.pF, self.mF = getMDPFrequencies(env,self.policyDistributions,meanPower=1000,digit_precision=7)

        self.frequencythreshold = 0.5  # 1./(self.time)
        self.policyFrequencies = (self.pF > self.frequencythreshold) * self.pF
        self.minFrequencies = np.amin(self.pF,axis=(2,3), where=self.pF>self.frequencythreshold, initial=np.inf)

        self.rewardStd = 0.1#env.rewardStd
        self.clear(self.mdp)

    def kl(self, x, y):
        return (y - x) ** 2 / (2 * self.rewardStd ** 2)

    def randmax(self, A):
        maxValue = max(A)
        index = [i for i in range(len(A)) if A[i] == maxValue]
        return np.random.choice(index)

    def playPolicy(self, mdp, policy, state):
        distribution = [self.policyDistributions[policy][state, a] for a in range(self.nA)]
        return np.random.choice(self.actions, 1, p=distribution)[0]

    def reset(self, s):
        self.clear(self.mdp)
        s

    def clear(self, mdp):
        self.time = 0
        self.startingState = mdp.s
        self.visitedStates = [self.startingState]
        self.tau = [0]
        self.newEpisode = False
        self.policy = 0
        self.test = [[True if self.policyFrequencies[self.startingState, self.policy, s, a] == 0 else False for a in
                      range(self.nA)] for s in range(self.nS)]
        self.nbDraws = np.zeros((self.nS, self.nA))
        self.cumRewards = np.zeros((self.nS, self.nA))
        self.means = np.zeros((self.nS, self.nA))
        self.gains = np.zeros(self.nP)
        self.maxGains = 0
        self.policyPulls = np.zeros(self.nP)
        self.indexes = np.zeros(self.nP)
        self.minIndexes = min(self.indexes)

    def play(self, state):
        return self.playPolicy(self.mdp, self.policy, state)

    def update(self, state, action, reward, newState):
        mdp = self.mdp
        self.time = self.time + 1
        self.cumRewards[state, action] = self.cumRewards[state, action] + reward
        self.nbDraws[state, action] = self.nbDraws[state, action] + 1
        self.means[state, action] = self.cumRewards[state, action] / self.nbDraws[state, action]

        # self.policyPulls[self.policy] = self.policyPulls[self.policy] + 1

        self.test[state][action] = True
        self.newEpisode = min(list(chain.from_iterable(self.test)))

        # if newState in self.visitedStates:
        #   self.newEpisode = True
        # else:
        #   self.visitedStates = self.visitedStates + [newState]

        if self.newEpisode:
            self.newEpisode = False
            self.startingState = [newState]
            self.tau = self.tau + [self.time - 1]


            self.frequencythreshold = min(0.5,1./np.sqrt(self.time))
            self.policyFrequencies = (self.pF > self.frequencythreshold) * self.pF
            self.minFrequencies = np.amin(self.pF,axis=(2,3), where=self.pF>self.frequencythreshold, initial=np.inf)

            self.gains = []
            for p in range(self.nP):
                gain = 0
                for s in range(self.nS):
                    for a in range(self.nA):
                        gain = gain + self.policyFrequencies[newState, p, s, a] * self.means[s, a]
                self.gains = self.gains + [gain]
            self.maxGains = max(self.gains)

            self.policyPulls = []
            for p in range(self.nP):
                pulls = self.time#np.Inf
                # pulls = 0
                for s in range(self.nS):
                    for a in range(self.nA):
                        if self.policyFrequencies[newState, p, s, a] > 0:
                            pulls = min(pulls, self.nbDraws[s, a])
                        # pulls = pulls + mdp.policyFrequencies[newState,p,s,a]*self.nbDraws[s,a]
                self.policyPulls = self.policyPulls + [pulls]

            # self.indexes = [self.policyPulls[p] * self.kl(self.gains[p], self.maxGains) + log(self.policyPulls[p]) if
            #                 self.policyPulls[p] > 0 else -np.Inf for p in range(self.nP)]

            self.indexes = [np.random.normal(self.gains[p], scale=0.5/np.sqrt(self.policyPulls[p])) if
                            self.policyPulls[p] > 0 else +np.Inf for p in range(self.nP)]

            self.minIndexes = max(self.indexes)

            self.policy = self.randmax(
                [self.minFrequencies[newState, p] if self.indexes[p] == self.minIndexes else -np.Inf for p in
                 range(self.nP)])

            self.test = [[True if self.policyFrequencies[self.startingState, self.policy, s, a] == 0 else False for a in
                          range(self.nA)] for s in range(self.nS)]

    def name(self):
        return "TS-KTP"

def run_one_xp(env, learner, nbr_step):
    s = env.reset()
    learner.reset(s)
    average_cumulated_reward = np.zeros(nbr_step)
    current_average_reward = 0

    for t in range(nbr_step):
        state = s
        action = learner.play(state)
        s, r, _, i = env.step(action)
        learner.update(state, action, r, s)

        average_cumulated_reward[t] = (t*current_average_reward + r) / (t+1)
        current_average_reward = average_cumulated_reward[t]

    return average_cumulated_reward

def run_multiple_xp(env, learner, nbr_exp, nbr_step ):
    rewards = np.zeros((nbr_exp, nbr_step))
    for exp in range(nbr_exp):
        s = env.reset()
        learner.reset(s)
        average_cumulated_reward = np.zeros(nbr_step)
        current_average_reward = 0

        for t in range(nbr_step):
            state = s
            action = learner.play(state)
            s, r, _, i = env.step(action)
            learner.update(state, action, r, s)

            average_cumulated_reward[t] = (t*current_average_reward + r) / (t+1)
            current_average_reward = average_cumulated_reward[t]

            rewards[exp,t] = current_average_reward

    return np.mean(rewards,0)

# ## Experiments
#
#
# STATES = len(states)
# ACTIONS = len(actions)
# nEnv = 1
# nExp = 100
# ITER = 1500
#
# def envF():
#     return randomMDP([randomPolicyDistribution(states,actions)] + dPolicyDistributions(states, actions))
#
#
# def learnerF(mdp=None):
#     return [UCRL3KTP.UCRL3KTP(mdp.nS, mdp.nA, mdp, 0.5), PSRLKTP.PSRLKTP(mdp.nS, mdp.nA, mdp, 0.5), IMEDKTP(mdp.nS, mdp.nA, mdp, 0.5)]
#
#
#
#
# def run_bayesian_xp(envF, nLearners, learnerF, nbr_env, nbr_exp, nbr_step ):
#     bayesians = [np.zeros((nbr_env, nbr_exp, nbr_step)) for l in range(nLearners)]
#     maxGains = np.zeros(nbr_env)
#     for envNumber in range(nbr_env):
#         env = envF()
#         maxGains[envNumber] = np.mean(env.maxPolicyGains)
#         learners = learnerF(mdp=env)
#         for l in range(nLearners):
#             learner = learners[l]
#
#             for exp in range(nbr_exp):
#                 s = env.reset()
#                 learner.reset(s)
#                 average_cumulated_reward = np.zeros(nbr_step)
#                 current_average_reward = 0
#
#                 for t in range(nbr_step):
#                     state = s
#                     action = learner.play(state)
#                     s, r, _, i = env.step(action)
#                     learner.update(state, action, r, s)
#
#                     average_cumulated_reward[t] = (t*current_average_reward + r) / (t+1)
#                     current_average_reward = average_cumulated_reward[t]
#
#                     bayesians[l][envNumber, exp, t] = current_average_reward
#
#
#
#     return bayesians +  [np.mean(maxGains)]
#
# ar_ucrl3, ar_psrl, ar_imedrl, maxGains = run_bayesian_xp(envF, 3, learnerF, nEnv, nExp, ITER)
# ## Save
# data = np.array([ar_ucrl3, ar_psrl, ar_imedrl])
# file = f"ktpRewards{STATES}S{ACTIONS}A{nEnv}E{nExp}RunsH{ITER}.npy"
# np.save('/home/saber/UCRLKTP/'+file, data)
#
# arrayMaxGains = np.array(maxGains)
# file = f"ktpMaxGains{STATES}S{ACTIONS}A{nEnv}E{nExp}RunsH{ITER}.npy"
# np.save('/home/saber/UCRLKTP/'+file, arrayMaxGains)
#
# ## Plots
#
#
# #np.set_printoptions(4)
# #print(mdp.maxPolicyGains)
#
# plt.clf()
# plt.figure(figsize=(15, 8))
# plt.title(f"Average Reward : nS={STATES}, nA={ACTIONS}, nEnv={nEnv}, runsPerEnv={nExp}, horizon={ITER}")
# plt.plot(np.mean(ar_imedrl,(0,1)), label=f"IMED-KTP - {np.mean(ar_imedrl,(0,1))[-1]:.3f}", color = 'b')
# plt.plot(np.quantile(ar_imedrl, 0.95, (0,1)), linestyle=':',linewidth=0.7, color='b')
# plt.plot(np.quantile(ar_imedrl, 0.05, (0,1)), linestyle=':',linewidth=0.7, color='b')
# plt.plot(np.mean(ar_ucrl3,(0,1)), label=f"UCRL3-KTP - {np.mean(ar_ucrl3,(0,1))[-1]:.3f}", color = 'r')
# plt.plot(np.quantile(ar_ucrl3, 0.95, (0,1)), linestyle=':',linewidth=0.7, color='r')
# plt.plot(np.quantile(ar_ucrl3, 0.05, (0,1)), linestyle=':',linewidth=0.7, color='r')
# plt.plot(np.mean(ar_psrl,(0,1)), label=f"PSRL-KTP - {np.mean(ar_psrl,(0,1))[-1]:.3f}", color = 'orange')
# plt.plot(np.quantile(ar_psrl, 0.95, (0,1)), linestyle=':',linewidth=0.7, color='orange')
# plt.plot(np.quantile(ar_psrl, 0.05, (0,1)), linestyle=':',linewidth=0.7, color='orange')
# plt.legend()
# file = f"ktpRewards{STATES}S{ACTIONS}A{nEnv}E{nExp}RunsH{ITER}.png"
# plt.savefig('/home/saber/UCRLKTP/'+file)
#
# plt.clf()
# plt.figure(figsize=(15, 8))
# plt.title(f"Average Regret : nS={STATES}, nA={ACTIONS}, nEnv={nEnv}, runsPerEnv={nExp}, horizon={ITER}")
# plt.plot(maxGains - np.mean(ar_imedrl,(0,1)), label=f"IMED-KTP", color = 'b')
# plt.plot(np.quantile(maxGains - ar_imedrl, 0.95, (0,1)), linestyle=':',linewidth=0.7, color='b')
# plt.plot(np.quantile(maxGains - ar_imedrl, 0.05, (0,1)), linestyle=':',linewidth=0.7, color='b')
# plt.plot(maxGains - np.mean(ar_ucrl3,(0,1)), label=f"UCRL3-KTP", color = 'r')
# plt.plot(np.quantile(maxGains - ar_ucrl3, 0.95, (0,1)), linestyle=':',linewidth=0.7, color='r')
# plt.plot(np.quantile(maxGains - ar_ucrl3, 0.05, (0,1)), linestyle=':',linewidth=0.7, color='r')
# plt.plot(maxGains - np.mean(ar_psrl,(0,1)), label=f"PSRL-KTP", color ='orange')
# plt.plot(np.quantile(maxGains - ar_psrl, 0.95, (0,1)), linestyle=':',linewidth=0.7, color='orange')
# plt.plot(np.quantile(maxGains - ar_psrl, 0.05, (0,1)), linestyle=':',linewidth=0.7, color='orange')
# plt.plot([0 for t in range(ITER)], color = 'k' )
# plt.legend()
# file = f"ktpRegrets{STATES}S{ACTIONS}A{nEnv}E{nExp}RunsH{ITER}.png"
# plt.savefig('/home/saber/UCRLKTP/'+file)
# plt.show()