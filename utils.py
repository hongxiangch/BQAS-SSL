import pickle
import datetime
import torch

def load_pkl(input_file):  # loading pkl
    f = open(input_file, 'rb')
    output_file = pickle.load(f)
    f.close()
    return output_file


def save_pkl(data, loc):
    f = open(loc, 'wb')
    pickle.dump(data, file=f)
    f.close()
    return 0

#
def getTime():
    year = str(datetime.datetime.today().year)
    month = str(datetime.datetime.today().month)
    if int(month) < 10:
        month = '0'+month
    day = str(datetime.datetime.today().day)
    if int(day) < 10:
        day = '0' + day
    hour = str(datetime.datetime.today().hour)
    if int(hour) < 10:
        hour = '0' + hour
    minute = str(datetime.datetime.today().minute)
    if int(minute) < 10:
        minute = '0' + minute
    second = str(datetime.datetime.today().second)
    if int(second) < 10:
        second = '0' + second

    return year+month+day+hour+minute+second


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2