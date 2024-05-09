import warnings
warnings.simplefilter('ignore', FutureWarning)
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

result_single = "./tk"
result_multi  = "./tk_sg_km_data-all"
input_dir = "./input"

n_fft=2048
win_length = n_fft
hop_length = win_length // 4
window = 'hann'
num_mels = 256
sr = 44100

mmin = -80
mmax = 0

def mel(y):
    mel = librosa.feature.melspectrogram(y=y,sr=44100,n_mels=256,n_fft=2048,hop_length=512, win_length=2048, window='hann')
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel

def plot(s, o1, o2, p):
    ms = mel(s)
    m1 = mel(o1)
    m2 = mel(o2)

    plt.rcParams["font.size"] = 12

    ## SRC
    fig = plt.figure(figsize=(13,10))
    # waveform
    ax = fig.add_subplot(8,1,(1,2),xlabel="time [sec]")
    ax.set(ylim=[-1.0,1.0])
    ax.set(xlim=[0, len(s)/sr])
    librosa.display.waveshow(s, sr=sr, x_axis='s')
    ax.tick_params(bottom=False, top=False, left=True, right=False, labelleft=True, labelright=False)
    ax.axes.xaxis.set_visible(False)
    # mel
    ax = fig.add_subplot(8,1,(3,8),xlabel="time [sec]")
    librosa.display.specshow(ms, sr=sr, x_axis='s', y_axis='mel')
    plt.xticks( np.arange(0, len(s)/sr, 0.5))
    plt.clim(mmin, mmax)
    # save fig
    plt.tight_layout()
    fig.savefig("../pics/input/%s"%p, format="png", dpi = 150)
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(13,10))
    # waveform
    ax = fig.add_subplot(8,1,(1,2),xlabel="time [sec]")
    ax.set(ylim=[-1.0,1.0])
    ax.set(xlim=[0, len(o1)/sr])
    librosa.display.waveshow(o1, sr=sr, x_axis='s')
    ax.tick_params(bottom=False, top=False, left=True, right=False, labelleft=True, labelright=False)
    ax.axes.xaxis.set_visible(False)
    # mel
    ax = fig.add_subplot(8,1,(3,8),xlabel="time [sec]")
    librosa.display.specshow(m1, sr=sr, x_axis='s', y_axis='mel')
    plt.xticks( np.arange(0, len(o1)/sr, 0.5))
    plt.clim(mmin, mmax)
    # save fig
    plt.tight_layout()
    fig.savefig("../pics/tk/%s"%p, format="png", dpi = 150)
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(13,10))
    # waveform
    ax = fig.add_subplot(8,1,(1,2),xlabel="time [sec]")
    ax.set(ylim=[-1.0,1.0])
    ax.set(xlim=[0, len(o2)/sr])
    librosa.display.waveshow(o2, sr=sr, x_axis='s')
    ax.tick_params(bottom=False, top=False, left=True, right=False, labelleft=True, labelright=False)
    ax.axes.xaxis.set_visible(False)
    # mel
    ax = fig.add_subplot(8,1,(3,8),xlabel="time [sec]")
    librosa.display.specshow(m2, sr=sr, x_axis='s', y_axis='mel')
    plt.xticks( np.arange(0, len(o2)/sr, 0.5))
    plt.clim(mmin, mmax)
    # save fig
    plt.tight_layout()
    fig.savefig("../pics/tk_sg_km_data-all/%s"%p, format="png", dpi = 150)
    plt.clf()
    plt.close()

def main():
    for i in range(1, 11):
        i1, _ = librosa.load(os.path.join(input_dir,  "%d_src_tk_1.wav"%i), sr=44100)
        o1, _ = librosa.load(os.path.join(result_single, "%d_out_tk_1.wav"%i), sr=44100)
        o2, _ = librosa.load(os.path.join(result_multi, "%d_out_tk_1.wav"%i), sr=44100)
        if len(o1) > len(o2):
            i1 = np.append(i1, np.zeros(len(o1)-len(i1)))
            o2 = np.append(o2, np.zeros(len(o1)-len(o2)))
        else:
            i1 = np.append(i1, np.zeros(len(o2)-len(i1)))
            o1 = np.append(o1, np.zeros(len(o2)-len(o1)))
        # print(len(i1), len(o1), len(o2))
        outpath = "%d_tk_1.png"%i
        plot(i1, o1, o2, outpath)
        for name in ["ki", "kz", "sg"]:
            for j in range(1,6):
                i2, _ = librosa.load(os.path.join(input_dir,  "%d_src_%s_%d.wav"%(i, name, j)), sr=44100)
                o3, _ = librosa.load(os.path.join(result_single, "%d_out_%s_%d.wav"%(i, name, j)), sr=44100)
                o4, _ = librosa.load(os.path.join(result_multi, "%d_out_%s_%d.wav"%(i, name, j)), sr=44100)
                if len(i2) < len(o1):
                        i2 = np.append(i2, np.zeros(len(o1)-len(i2)))
                        o3 = np.append(o3, np.zeros(len(o1)-len(o3)))
                        o4 = np.append(o4, np.zeros(len(o1)-len(o4)))
                else:
                    i2 = i2[:len(o1)]
                    o3 = o3[:len(o1)]
                    o4 = o4[:len(o1)]
                # print(len(i2), len(o3), len(o4))
                outpath = "%d_%s_%d.png"%(i, name, j)
                plot(i2, o3, o4, outpath)


if __name__ == "__main__":
    main()