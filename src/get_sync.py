# Main sync eval
import syncmetrics as syncm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing 
import novainstrumentation as ni
import json

# Load config and signals
with open("./config.json") as conf_file:  
    conf = json.load(conf_file)

# json.dumps(conf, indent=4)
# print(conf)

srate = conf["srate"]
filepath = conf["AcqPath"]
txtload = True
ch_offset = 0
chann_a = conf["chann_a"]
chann_b = conf["chann_b"]
if txtload and conf["Platform"]== "bitalino":
    ch_offset = 4
    chann_a = chann_a + ch_offset
    chann_b = chann_b + ch_offset

raw_A, rawtime, srate = syncm.loadsignal(filepath, chann_a, txtload, srate)
raw_B, rawtime, srate = syncm.loadsignal(filepath, chann_b, txtload, srate)

#Signal clipping
#startpoint = 70*srate
#endpoint = -10*srate
startpoint = 0
endpoint = len(raw_A)
signal_A = raw_A[startpoint:endpoint]
signal_B = raw_B[startpoint:endpoint]
smooth_A = syncm.smoothing(signal_A)
smooth_B = syncm.smoothing(signal_B)
time = rawtime[startpoint:endpoint]
t_offset = time[0]
time = time - t_offset
sampletime = np.arange(len(signal_A))

# Plot offset
i_offset = 1


text_list_of_metrics = [ "lin_reg_r",
                         "inst_phase_difference",
                         "MPC",
                         "MSC",
                         "similar der_smooth",    # similar derivative sm
                         "similar der",           # similar derivative
                         "relative_int",          # relative area B/A (integral)
                         "normdiff",              # normalised diff
                         "cosine_similarity"]     # cosine similarity (reshape needed)                         
#                         "correlation_coeff",    # correlation coefficient 
#                         "cross_corr_max",

                           

list_of_metrics_slw = [syncm.lin_reg_r_metric,
                      syncm.inst_phase_difference,
                      syncm.MPC,
                      syncm.MSC,
                      syncm.similar_der_smooth,
                      syncm.similar_der,
                      syncm.relative_int,
                      syncm.normdiff,
                      syncm.cos_similarity
#                      syncm.correlation_coeff,
#                      syncm.cross_corr_max,

                      ]

#text_list_of_metrics = ["cross_corr_max"]
#list_of_metrics_slw = [syncm.cross_corr_max]


###############################################################################
def get_delay(a,b,onset_a,onset_b):
    t_event=[]
    min_len=np.min([len(onset_a),len(onset_b)])
    if  np.size(onset_a) > min_len:
        onset_a=onset_a[0:min_len]
    if  np.size(onset_b)>min_len:
        onset_b=onset_b[0:min_len]
          
    delay=abs(np.array(onset_a)-np.array(onset_b))
    amp_peak_a=a[np.array(onset_a)];
    amp_peak_b=b[np.array(onset_b)];
    peak_diff=abs(np.array(amp_peak_a)-np.array(amp_peak_b))
    
    for i in range(min_len):
        sign=np.array(onset_a[i])-np.array(onset_b[i])
        if sign>0:
            t_event +=[onset_a[i]]
        else:
            t_event +=[onset_b[i]]
    return t_event, delay, peak_diff

# Plot
plotdim = 6
fig, axs = plt.subplots(plotdim,2, facecolor='w', edgecolor='k')


# SHADI's neurokit RSP peak detect
detect_peaks = False
if detect_peaks:
    rsp_onset_A = syncm.rsp_peak_detect(signal_A, srate, 1.6)
    rsp_onset_B = syncm.rsp_peak_detect(signal_B, srate, 1.6)
#    plt.plot(time, signal_A, '-gD', markevery=rsp_onset_A, marker='o', color='b')
#    plt.plot(time, signal_B, '-gD', markevery=rsp_onset_B, marker='o', color='g')
    rsp_time_A = np.diff(np.array(rsp_onset_A), axis=0)
    rsp_time_B = np.diff(np.array(rsp_onset_B), axis=0)
    i_offset = 3

    axs[0,0].plot(time,signal_A,'-gD',markevery=rsp_onset_A,marker='o', color='b')
    axs[0,0].plot(time,signal_B,'-gD',markevery=  rsp_onset_B,marker='o', color='g')
    axs[0,0].set_title('Peak detection in signals')

    TE, D, PD = get_delay(signal_A, signal_B, rsp_onset_A, rsp_onset_B)

    axs[1,0].stem(TE,D) 
    axs[1,0].set_title('Time Delay of peaks')
    axs[2,0].stem(TE,PD)
    axs[2,0].set_title('Amp Delay of peaks')
    
else:
    axs[0,0].plot(time,signal_A, color='b')
    axs[0,0].plot(time,signal_B, color='g')
    axs[0,0].set_title('Signals')


i=0

#for metric in list_of_metrics_slw[:j]:
wind_val = 12000
for metric in list_of_metrics_slw:
    print(metric)

    # Use smooth
#    m,t = syncm.compute_metric_slw(smooth_A,smooth_B,metric, wind_val, overlap=.9)  ## window=win , half win, double win

    # Use signal
    m,t = syncm.compute_metric_slw(signal_A,signal_B,metric, wind_val, overlap=.9)  ## window=win , half win, double win

    # Plot and color
    axs[int((i + i_offset) % plotdim),int((i + i_offset)/plotdim)].plot(t, m, str('C' + str((i+1))))
    axs[int((i + i_offset) % plotdim),int((i + i_offset)/plotdim)].set_title(text_list_of_metrics[i])
    i=i+1

fig.set_size_inches(14, 9)
plt.tight_layout()
figname = "../results/" + conf["AcqPath"][-conf["AcqPath"][::-1].find('/'):] + '_w{:05d}.png'.format(wind_val)
fig.savefig(figname)


### SINGLE PLOTS
#
#plt.plot(syncm.comp_inst_phase(smooth_A))
#plt.plot(syncm.comp_inst_phase(smooth_B))
#
#
#plt.plot(syncm.comp_inst_phase(smooth_A[:10000]))
#plt.plot(syncm.comp_inst_phase(smooth_B[:10000]))
#
#plt.plot(np.diff(syncm.comp_inst_phase(smooth_A[:10000])))
#plt.plot(np.diff(syncm.comp_inst_phase(smooth_B[:10000])))
#
#syncm.lin_reg_r_metric(smooth_A,smooth_B)
