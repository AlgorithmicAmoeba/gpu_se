import pandas
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')

black = '#2B2B2D'
red = '#E90039'
orange = '#FF1800'
white = '#FFFFFF'
yellow = '#FF9900'

def plot_speedup():
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 16, 'text.color': white, 'axes.labelcolor': white,
                         'axes.edgecolor': white, 'xtick.color': white, 'ytick.color': white})

    plt.gca().set_facecolor(black)

    predict_df = pandas.read_csv('PF_predict.csv', index_col=0)
    predict_df['speedup'] = predict_df['CPU'] / predict_df['GPU']
    plt.semilogy(predict_df.index, predict_df['speedup'], '.', color=yellow)

    update_df = pandas.read_csv('PF_update.csv', index_col=0)
    update_df['speedup'] = update_df['CPU'] / update_df['GPU']
    plt.semilogy(update_df.index, update_df['speedup'], 'x', color=red)

    resample_df = pandas.read_csv('PF_resample.csv', index_col=0)
    resample_df['speedup'] = resample_df['CPU'] / resample_df['GPU']
    # plt.semilogy(resample_df.index, resample_df['speedup'], 'g^')

    # plt.legend(['Predict', 'Update', 'Resample'])
    plt.legend(['Predict', 'Update'], facecolor=black)
    # plt.title('Speed-up of particle filter')
    plt.ylabel('Speed-up')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.axhline(1, color=white, alpha=0.4)
    plt.tight_layout()

    plt.savefig('PF_speedup_pretty.png', transparent=True)
    plt.show()


def plot_times():
    predict_df = pandas.read_csv('PF_predict.csv', index_col=0)
    update_df = pandas.read_csv('PF_update.csv', index_col=0)
    resample_df = pandas.read_csv('PF_resample.csv', index_col=0)

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 16, 'text.color': white, 'axes.labelcolor': white,
                         'axes.edgecolor': white, 'xtick.color': white, 'ytick.color': white})

    plt.subplot(1, 2, 1)
    plt.gca().set_facecolor(black)
    # plt.semilogy(resample_df.index, resample_df['CPU'], 'g^')
    plt.semilogy(predict_df.index, predict_df['CPU'], '.', color=yellow)
    plt.semilogy(update_df.index, update_df['CPU'], 'x', color=red)

    # plt.legend(['Resample', 'Predict', 'Update'])
    plt.legend(['Predict', 'Update'], facecolor=black)
    plt.ylabel('Time (s)')
    plt.xlabel('$ \log_2(N) $ particles')
    # plt.title('CPU')

    plt.subplot(1, 2, 2, sharey=plt.gca())
    plt.gca().set_facecolor(black)
    # plt.semilogy(resample_df.index, resample_df['GPU'], 'g^')
    plt.semilogy(predict_df.index, predict_df['GPU'], '.', color=yellow)
    plt.semilogy(update_df.index, update_df['GPU'], 'x', color=red)

    # plt.legend(['Resample', 'Predict', 'Update'])
    plt.legend(['Predict', 'Update'], facecolor=black)
    plt.ylabel('Time (s)')
    plt.xlabel('$ \log_2(N) $ particles')
    # plt.title('GPU')

    # plt.suptitle('Run times particle filter methods')
    plt.tight_layout()

    plt.savefig('PF_times_pretty.png', transparent=True)
    plt.show()


plot_times()
plot_speedup()
