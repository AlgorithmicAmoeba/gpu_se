import pandas
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')


def plot_speedup():
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 12})

    predict_df = pandas.read_csv('PF_predict.csv', index_col=0)
    predict_df['speedup'] = predict_df['CPU'] / predict_df['GPU']
    plt.semilogy(predict_df.index, predict_df['speedup'], 'b.')

    update_df = pandas.read_csv('PF_update.csv', index_col=0)
    update_df['speedup'] = update_df['CPU'] / update_df['GPU']
    plt.semilogy(update_df.index, update_df['speedup'], 'rx')

    resample_df = pandas.read_csv('PF_resample.csv', index_col=0)
    resample_df['speedup'] = resample_df['CPU'] / resample_df['GPU']
    plt.semilogy(resample_df.index, resample_df['speedup'], 'g^')

    plt.legend(['Predict', 'Update', 'Resample'])
    plt.title('Speed-up of particle filter')
    plt.ylabel('Speed-up')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.axhline(1, color='black', alpha=0.4)
    plt.tight_layout()

    plt.savefig('PF_speedup.pdf')
    plt.show()


def plot_times():
    predict_df = pandas.read_csv('PF_predict.csv', index_col=0)
    update_df = pandas.read_csv('PF_update.csv', index_col=0)
    resample_df = pandas.read_csv('PF_resample.csv', index_col=0)

    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 12})

    plt.subplot(1, 2, 1)
    plt.semilogy(predict_df.index, predict_df['CPU'], 'b.')
    plt.semilogy(update_df.index, update_df['CPU'], 'rx')
    plt.semilogy(resample_df.index, resample_df['CPU'], 'g^')

    plt.legend(['Predict', 'Update', 'Resample'])
    plt.ylabel('Time (s)')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.title('CPU')

    plt.subplot(1, 2, 2, sharey=plt.gca())
    plt.semilogy(predict_df.index, predict_df['GPU'], 'b.')
    plt.semilogy(update_df.index, update_df['GPU'], 'rx')
    plt.semilogy(resample_df.index, resample_df['GPU'], 'g^')

    plt.legend(['Predict', 'Update', 'Resample'])
    plt.ylabel('Time (s)')
    plt.xlabel('$ \log_2(N) $ particles')
    plt.title('GPU')

    plt.suptitle('Run times particle filter methods')
    plt.tight_layout()

    plt.savefig('PF_times.pdf')
    plt.show()


plot_times()
# plot_speedup()
