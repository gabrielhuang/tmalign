from copy import deepcopy
import numpy as np
import scipy.stats


class Distribution(object):
    def get_log_pdf(self, x):
        pass
    
    def get_pdf(self, x):
        return np.exp(self.get_log_prob(x))
    
    def __getitem__(self, x):
        return self.get_log_pdf(x)
    
    
class ObservationModel(Distribution): 
    pass
    
    
class DurationModel(Distribution):
    def get_log_survival(self, x):
        pass
    
    def get_survival(self, x):
        return np.exp(self.get_log_survival(x))
    

class UnimodalGaussian(ObservationModel):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def get_log_pdf(self, x):
        return scipy.stats.norm.logpdf(x, self.mean, self.std)

    
class PoissonDM(DurationModel):
    def __init__(self, mean):
        self.mean = mean
        
    def get_log_pdf(self, x):
        return scipy.stats.poisson.logpmf(x, self.mean)
    
    def get_log_survival(self, x):
        return np.log(1. - scipy.stats.poisson.cdf(x-1, self.mean))

                      
class GeometricDM(DurationModel):
    '''
    Geometric Distribution starting at 0
        P(x) = p(1-p)^x
    with mean
        E[x] = (1-p) / p
    '''
    def __init__(self, mean):
        self.mean = mean
        self.p = 1. / (1 + self.mean)
    
    def get_log_pdf(self, x):
        return scipy.stats.geom.logpmf(x, self.p, loc=-1)
    
    def get_log_survival(self, x):
        return np.log(1. - scipy.stats.geom.cdf(x-1, self.p, loc=-1))
    
    
def log_sum_exp(args):
    '''
    Compute the log of sum of exp of args intelligently.
    
    log(sum_i exp(x_i)) = x_j + log(sum_i exp(x_i-x_j))
    '''
    args = np.asarray(list(args))
    largest = np.max(args)
    if largest <= float('-inf'): # avoid nan
        return float('-inf')
    return largest + np.log(np.sum(np.exp(args-largest)))


def l1_normalized(args, axis=None):
    '''
    Compute the log of sum of exp of args intelligently, then divide by it.
    '''
    if axis == 1:
        new_args = []
        for row in args:
            new_args.append(l1_normalized(row))
        return np.asarray(new_args)
    elif axis is not None:
        raise Exception('Only axis 1 is supported')
    args = np.asarray(list(args))
    log_sum = log_sum_exp(args)
    return args - log_sum


class HMM(object):
    '''
    Linear-Chain hidden Markov Model
    '''
    def __init__(self, durations, obs_model, frames_per_beat):
        '''
        if bpm is None, use NO tempo model, e.g., 
        
        '''
        self.logobs = self.obs_model = obs_model
        self.frames_per_beat = frames_per_beat  # beat to frames
        # geometric distribution
        self.exits = map(self.get_exit, durations)
        
    def nstates(self):
        return len(self.obs_model)
    
    def states(self):
        return range(self.nstates())
    
    def get_exit(self, duration_beats):
        '''
        Return log probability of leaving a state.
        
        Implicit duration model is a geometric distribution
            d(u) = p(1-p)^u
        
        We want E[d(u)] = (1-p)/p = duration,
        which corresponds to p = 1 / (d+1)
        '''
        return 1. / (duration_beats * self.frames_per_beat + 1)
        
    def logtrans(self, prestate, state):
        if prestate == state:
            return np.log(1 - self.exits[prestate])
        elif prestate+1 == state:
            return np.log(self.exits[prestate])
        else:  # cannot skip state
            return np.log(0.)
        
    def decode(self, obs, mode='offline-viterbi', verbose=False):
        '''
        Viterbi Algorithm or "Online Viterbi"
        
        Parameters
        ----------
        obs : list of observations
            observed variables x_0 .. x_{T-1}
        mode : string
            if online == 'offline-viterbi'
                Compute argmax_{z_0 .. z_{T-1}} P(z_0 .. z_{T-1} | x_0 .. x_{T-1})
            elif online == 'online-viterbi'
                Compute argmax_{z_t} max_{z_0 .. z_{t-1}} P(z_t, z_0 .. z_{t-1} | x_0 .. x_t) for all 0 <= t <= T
            elif online == 'online-forward':
                Compute argmax_{z_t}P(z_t | x_0 .. x_t) for all 0 <= t <= T
        '''
        if mode not in ('offline-viterbi', 'online-viterbi', 'online-forward'):
            raise ValueError('Unsupported mode {}'.format(mode))
        
        # Initialize
        alpha = np.zeros((len(obs), self.nstates()))
        path = {}
        self.alpha = alpha # debug
        self.path = path # debug
        for state in self.states():
            alpha[0, state] = self.logobs[state][obs[0]] + np.log(1. if state==0 else 0.)
            path[state] = [state]
        # Forward Recursion
        for t, x in list(enumerate(obs))[1:]:
            if verbose and t%10==0: print '{}/{}'.format(t, len(obs))
            old_path = deepcopy(path)
            for state in self.states():
                prestate_scores = [self.logtrans(prestate, state) + alpha[t-1][prestate]
                    for prestate in self.states() if prestate==state or prestate==state-1]

                if mode == 'online-forward':
                    alpha[t, state] = self.logobs[state][x] + log_sum_exp(prestate_scores)
                    
                else: # mode in('offline-viterbi', 'online-viterbi'):
                    best_prestate = np.argmax(prestate_scores)
                    actual_prestate = state+1-len(prestate_scores)+best_prestate  # correct index
                    alpha[t, state] = (self.logobs[state][x]) + prestate_scores[best_prestate]
                    path[state] = old_path[actual_prestate] + [state]
        
        # Return best path
        if mode == 'offline-viterbi':
            best_last_state = self.nstates()-1  # has to finish on last state
            best_path = path[best_last_state]
            return best_path
        else:  # mode online
            best_online = np.argmax(alpha, axis=1)
            return best_online

        
class HSMM(HMM):
    '''
    Hidden Semi-Markov Model
    '''
    def __init__(self, obs_model, dur_model):
        '''
        if bpm is None, use NO tempo model, e.g., 
        
        '''
        self.obs_model = obs_model
        self.dur_model = dur_model

    def decode(self, obs, max_duration=None):
        '''
        Right-Censored Forward Algorithm (HERE ONLINE ONLY)
        
        Compute argmax_{z_t} P(z_t | x_0 .. x_t) at every frame t.
        
        Parameters
        ----------
        
        Notice that notations are 0-indexed:
        t = 0 .. T-1 are the frame indices
        j = 0 .. J-1 are the possible states (music events)
        x_0 .. x_{T-1} are the observations
        z_0 .. z_{T-1} are the corresponding states
        
        f[t, j] = P(z_t = j | x_0 .. x_t) is the filtered z_t
        f_out[t, j] = P(z_{t+1} != j, z_t = j | x_0 .. x_t) is the probability of leaving j at t+1
        
        '''
        # Initialize
        f = np.log(0) * np.ones((len(obs+1), 1+self.nstates()))
        f_out = np.log(0) * np.ones((len(obs+1), 1+self.nstates()))
        f_out[0, 0] = self.dur_model[0].get_log_pdf(1)  # f_out[0, 1] = occupancy(j, 1)
        # Special Cases:
        #     f_out[:, -1] = np.log(0.) # special case included
        #     f_out[-1, j] = np.log(0.) # when j!=0, special case included
        #     f_out[:, -1] = np.log(0.)  # special case included
        f_out[-1, 0] = 0. # special case
        
        self.f = f  # debug
        self.f_out = f_out  # debug
        
        # Precompute Observation probabilities (for optimization, not necessary)
        self.logobs = logobs = np.zeros((len(obs), self.nstates()))
        for t, x in list(enumerate(obs)):
            for j in self.states():
                logobs[t, j] = self.obs_model[j][x]
            
        # Forward recursion
        path = []  # only one path to track
        for t, x in list(enumerate(obs))[1:]:
            for j in self.states():
                f[t, j] = log_sum_exp([logobs[t-u+1:t+1, j].sum()
                                      + f_out[t-u, j-1]
                                      + self.dur_model[j].get_log_survival(u) 
                                       for u in xrange(1, t+2)
                                       if not max_duration or u<max_duration])
                f_out[t, j] = log_sum_exp([logobs[t-u+1:t+1, j].sum()
                                      + f_out[t-u, j-1]
                                      + self.dur_model[j].get_log_pdf(u) 
                                           for u in xrange(1, t+2)
                                           if not max_duration or u<max_duration])
        # Best path
        best_path = np.argmax(f, axis=1)
        return best_path