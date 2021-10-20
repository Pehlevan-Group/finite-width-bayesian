import pytz
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

tz = 'US/Eastern'

def time_diff(t_start):
    t_end = datetime.now(pytz.timezone(tz))
    t_diff = relativedelta(t_end, t_start)  # later/end time comes first!
    return '{d}d {h}h {m}m {s}s'.format(d=t_diff.days, h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def curr_time():
    return datetime.now(pytz.timezone(tz))


def convert_nt(tensor, conv_layer = None):
    
    if len(tensor.shape) != 6:
        return tensor
    elif conv_layer != None:
        if (conv_layer + 1) % 2 == 1:
            tensor = np.moveaxis(tensor, (2,3), (4,5))
    
    return np.moveaxis(tensor, (3,4),(4,3))