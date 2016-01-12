import numpy as np
import numbers
import matplotlib.pyplot as plt


ABC_TO_PITCHES_TABLE = {'A': 0, 'B': 2, 'C':3, 'D': 5,
                        'E': 7, 'F': 8, 'G': 10,
                        'a': 12, 'b': 14, 'c': 15, 'd': 17,
                        'e': 19, 'f': 20, 'g': 22}
STD=4.


def abc_to_pitches(abc, octave=4):
    '''
    Convert string of notes to MIDI pitches
    '''
    pitches = []
    for note in abc:
        rel_pos = ABC_TO_PITCHES_TABLE[note]
        pitches.append(23 + 12 * octave + rel_pos)
    return np.asarray(pitches)


def get_alignment(durations, bpm, frame_sec):
    '''
    Given the durations and tempi of successive notes,
    return at every time t: alignment z[t] (timestamp) and tempo[t]
    
    Parameters
    ----------
    durations : list of float
        duration of each note in beats (quarter notes)
    bpm : float or list of float
        tempo in beats per minute, at each note
    frame_sec : float, optional
        seconds per frame (default 0.1 seconds)
        
    Returns
    -------
    z : list of int
        alignment of 
        z[t] = j means that score event j is played at frame t (time t*frame_sec)
    realtime_bpm : list of float
        bpm at each frame
    '''
    if isinstance(bpm, numbers.Number):
        bpm = [bpm] * len(durations)
    z = []
    realtime_bpm = []
    for i, (duration, current_bpm) in enumerate(zip(durations, bpm)):
            beat_sec = 60. / current_bpm  # seconds per beat
            frames_per_beat = beat_sec / frame_sec
            realtime_duration = int(frames_per_beat * duration)
            z += [i] * realtime_duration
            realtime_bpm += [current_bpm] * realtime_duration
    return np.asarray(z), np.asarray(realtime_bpm)
    

def sample_gaussian_obs(pitches, z, std):
    '''
    Observations are unimodal Gaussians centered around pitch
    
    Parameters
    ----------
    pitches : list of int
        pitches of notes (following MIDI tables)
    '''
    x = pitches[z] + std * np.random.normal(size=len(z))
    return np.asarray(x)


class Benchmark(object):
    '''
    Defines an alignment problem
    '''
    pass


class PitchBenchmark(Benchmark):
    def __init__(self, pitches, durations, bpm, frame_sec, name='untitled'):
        self.pitches = pitches
        self.durations = durations
        self.events = zip(self.pitches, self.durations)
        self.bpm = bpm
        self.frame_sec = frame_sec
        self.name = name
        self.z, self.realtime_bpm = get_alignment(durations, bpm, frame_sec)
        
    def __repr__(self):
        l = ' '.join(map(str, zip(self.pitches, self.durations)))
        return str(l)
    
    def frames_per_beat(self):
        return 60. / (self.bpm * self.frame_sec)
    
        
class UnimodalGaussianBenchmark(PitchBenchmark):
    def __init__(self, pitches, durations, bpm, frame_sec, std=STD, name='untitled'):
        PitchBenchmark.__init__(self, pitches, durations, bpm, frame_sec, name=name)
        self.x = sample_gaussian_obs(pitches, self.z, std)

    def plot(self):
        plt.figure()
        plt.plot(self.pitches[self.z])
        plt.plot(self.x) 
        plt.title('Benchmark [{}] - Alignment and Observations'.format(self.name))
        plt.xlabel('Frame t')
        plt.ylabel('Observation x')
        plt.figure()
        plt.plot(self.realtime_bpm)
        plt.title('Benchmark [{}] - Tempo VS Frame'.format(self.name))
        plt.xlabel('Frame t')
        plt.ylabel('Tempo (BPM)')