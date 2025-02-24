import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pingouin import pairwise_corr

#DZ
phoneme_list = ['S', 'Z', 'C', 'DZ', 'SZ', 'RZ', 'CZ', 'SI', 'CI', 'DZI']
for phoneme in phoneme_list:
    data = pd.read_csv('D:\\!code\\phoneme_' + phoneme + '_features.csv', header=None)
    column_names = data.iloc[0,:]
    column_names_3DL = column_names[0:103]
    column_names_2DL = column_names[103:191]
    column_names_audio = column_names[191:215]
    column_names_2DR = column_names[215:303]
    column_names_3DR = column_names[303:406]
    # 3DL: 0:102
    # 2DL: 103:190
    # Audio: 191:214
    # 2DR: 215:302
    # 3DR: 303:405
    data_n = np.array(data)
    data_n = data_n[1:,:]

    data_n_3DL = data_n[:,0:103]
    data_n_2DL = data_n[:,103:191]
    data_n_audio = data_n[:,191:215]
    data_n_2DR = data_n[:,215:303]
    data_n_3DR = data_n[:,303:406]

    df_3DL = pd.DataFrame(data_n_3DL, columns=column_names_3DL)
    for col in df_3DL.columns[0:]:
        df_3DL[col] = pd.to_numeric(df_3DL[col], errors='coerce')

    df_2DL = pd.DataFrame(data_n_2DL, columns=column_names_2DL)
    for col in df_2DL.columns[0:]:
        df_2DL[col] = pd.to_numeric(df_2DL[col], errors='coerce')

    df_3DR = pd.DataFrame(data_n_3DR, columns=column_names_3DR)
    for col in df_3DR.columns[0:]:
        df_3DR[col] = pd.to_numeric(df_3DR[col], errors='coerce')

    df_2DR = pd.DataFrame(data_n_2DR, columns=column_names_2DR)
    for col in df_2DR.columns[0:]:
        df_2DR[col] = pd.to_numeric(df_2DR[col], errors='coerce')

    df_audio = pd.DataFrame(data_n_audio, columns=column_names_audio)
    for col in df_audio.columns[0:]:
        df_audio[col] = pd.to_numeric(df_audio[col], errors='coerce')

    df_3DL.reset_index(drop=True, inplace=True)
    df_audio.reset_index(drop=True, inplace=True)
    df_2DL.reset_index(drop=True, inplace=True)
    df_3DR.reset_index(drop=True, inplace=True)
    df_2DR.reset_index(drop=True, inplace=True)

    df_3DL_audio = [df_3DL, df_audio]
    df_3DL_audio_res = pd.concat(df_3DL_audio, axis=1)
    df_2DL_audio = [df_2DL, df_audio]
    df_2DL_audio_res = pd.concat(df_2DL_audio, axis=1)
    df_3DR_audio = [df_3DR, df_audio]
    df_3DR_audio_res = pd.concat(df_3DR_audio, axis=1)
    df_2DR_audio = [df_2DR, df_audio]
    df_2DR_audio_res = pd.concat(df_2DR_audio, axis=1)

    """    pcor_3DL = pairwise_corr(df_3DL_audio_res, method='spearman')
    pcor_3DL.to_csv('D:\\!code\\!corr_res\\phoneme_' + phoneme + '_3DL_correlations.csv')
    pcor_2DL = pairwise_corr(df_2DL_audio_res, method='spearman')
    pcor_2DL.to_csv('D:\\!code\\!corr_res\\phoneme_' + phoneme + '_2DL_correlations.csv')
    pcor_3DR = pairwise_corr(df_3DR_audio_res, method='spearman')
    pcor_3DR.to_csv('D:\\!code\\!corr_res\\phoneme_' + phoneme + '_3DR_correlations.csv')
    pcor_2DR = pairwise_corr(df_2DR_audio_res, method='spearman')
    pcor_2DR.to_csv('D:\\!code\\!corr_res\\phoneme_' + phoneme + '_2DR_correlations.csv')"""


    ###
    df_correlation = df_2DL_audio_res.corr(method='spearman')
    plt.close()
    plt.figure(figsize=(25, 25))

    df_correlation_roi = df_correlation.iloc[:-24-11, -24:-1]#[:-24, -24:-1]
    ax = sns.heatmap(df_correlation_roi.transpose(), cmap='coolwarm', square=True,
                     linewidths=0.5, xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.5})
    plt.xlabel('Cechy obrazowe', fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel('Cechy akustyczne', fontsize=15)  # y-axis label with fontsize
    plt.savefig('D:\\!code\\!corr_res\\newfig' + phoneme + '_2DL.png')

    df_correlation = df_2DR_audio_res.corr(method='spearman')
    plt.close()
    plt.figure(figsize=(25, 25))

    df_correlation_roi = df_correlation.iloc[:-24-11, -24:-1]#[:-24, -24:-1]
    ax = sns.heatmap(df_correlation_roi.transpose(), cmap='coolwarm', square=True,
                     linewidths=0.5, xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.5})
    plt.xlabel('Cechy obrazowe', fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel('Cechy akustyczne', fontsize=15)  # y-axis label with fontsize
    plt.savefig('D:\\!code\\!corr_res\\newfig' + phoneme + '_2DR.png')

    df_correlation = df_3DL_audio_res.corr(method='spearman')
    plt.close()
    plt.figure(figsize=(25, 25))

    df_correlation_roi = df_correlation.iloc[:-24-11-5, -24:-1]#[:-24, -24:-1]
    ax = sns.heatmap(df_correlation_roi.transpose(), cmap='coolwarm', square=True,
                     linewidths=0.5, xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.5})
    plt.xlabel('Cechy obrazowe', fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel('Cechy akustyczne', fontsize=15)  # y-axis label with fontsize
    plt.savefig('D:\\!code\\!corr_res\\newfig' + phoneme + '_3DL.png')

    df_correlation = df_3DR_audio_res.corr(method='spearman')
    plt.close()
    plt.figure(figsize=(25, 25))

    df_correlation_roi = df_correlation.iloc[:-24-11-5, -24:-1]#[:-24, -24:-1]
    ax = sns.heatmap(df_correlation_roi.transpose(), cmap='coolwarm', square=True,
                     linewidths=0.5, xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.5})
    plt.xlabel('Cechy obrazowe', fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel('Cechy akustyczne', fontsize=15)  # y-axis label with fontsize
    plt.savefig('D:\\!code\\!corr_res\\newfig' + phoneme + '_3DR.png')

    #pcor_audio = pairwise_corr(df_audio, method='spearman')
    #pcor_audio.to_csv('D:\\!code\\!corr_res\\phoneme_' + phoneme + '_audio_correlations.csv')


    """df = pd.DataFrame(data_n, columns = column_names)
    for col in  df.columns[0:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')"""

    """pcor = pairwise_corr(df, method='spearman')
    pcor.to_csv('D:\\!code\\phoneme_' + phoneme + '_correlations.csv')
    print(df)"""

"""df = df#df.iloc[:, 103:215]
mat_temp = []
sumnanstemp = []
check_nan_in_df = df.isnull()
for idx in range(0, df.shape[1]):
    sumNans = sum(check_nan_in_df.iloc[:,idx] == True)
    sumnanstemp.append(sumNans)
    if sumNans >= 150:
        mat_temp.append(idx)"""
"""mat_temp.append(126)
mat_temp.append(139)
mat_temp.append(238)
mat_temp.append(251)
mat_temp.append(339)"""





"""sumnanstemp = np.asarray(sumnanstemp)
df = df.drop(df.columns[mat_temp], axis=1)"""

"""df = df.drop(df.columns[310], axis=1)
df = df.drop(df.columns[235], axis=1)
df = df.drop(df.columns[136], axis=1)
df = df.drop(df.columns[49], axis=1)"""

"""videoL = df #df.iloc[:,103:214]
#audioL = df.iloc[:,205:228]
videoL = np.array(df)
videoL = np.array(videoL[1:-1, :], dtype='float')"""
#r,p = stats.spearmanr(videoL, nan_policy='propagate')

"""plt.scatter(np.array(videoL["GLRLMSRLGEL"]), np.array(videoL["MFCC4"]))
plt.plot(np.sort(np.array(videoL["GLRLMSRLGEL"])), np.sort(np.array(videoL["MFCC4"])), color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.pause(0.1)"""

#np.savetxt("p_" + phoneme + ".csv", p, delimiter=",")
#np.savetxt('r_' + phoneme + '.csv', r, delimiter=",")
"""rval = df.corr(method = 'spearman', nan_policy='omiit')
r_video = rval.iloc[117:228, 117:228]
sns.heatmap(r_video)"""

"""audio = np.array(df.iloc[:,205:228])
video =np.array(df.iloc[:,117:204])
res, p = stats.spearmanr(audio, video)"""

"""sns.heatmap(res[0:], annot = True)

plt.show()"""


