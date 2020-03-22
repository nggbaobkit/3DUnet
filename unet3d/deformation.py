import numpy as np

training_modalities = ("t1", "t1ce", "flair", "t2")
means = [1002.8655, 754.4905, 185.99873, 260.69287]
stds = [1173.731, 882.37555, 228.1578, 323.1486]


def add_gaussian_noise(data, noise_variance=(30, 30)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = np.random.uniform(noise_variance[0], noise_variance[1])

    # # adding the same noise to all modalities
    # for i in range(len(training_modalities)):
    #     data[0, i] += np.random.normal(0.0, variance, size=data[0, i].shape)

    # adding noise to only one modality
    data[0, 0] += np.random.normal(0.0, variance, size=data[0, 0].shape)  # t1
    # data[0, 1] += np.random.normal(0.0, variance, size=data[0, 1].shape)  # t1ce
    # data[0, 2] += np.random.normal(0.0, variance, size=data[0, 2].shape)  # flair
    # data[0, 3] += np.random.normal(0.0, variance, size=data[0, 3].shape)  # tt2


    # print("Finished adding gaussian noise...")
    # print("Max value after adding: {}".format(np.amax(data)))
    # print("Min value after adding: {}".format(np.amin(data)))
    return data


def reverse_z_normalization(data):
    # data shape is (1x4x128x128x128)
    for i in range(len(training_modalities)):
        data[0, i] *= stds[i]
        data[0, i] += means[i]
        if np.amax(data) < 900:
            print("Max in {}: {}".format(i, np.amax(data)))
            print("Min in {}: {}".format(i, np.amin(data)))
    # print("Finished reversing z normalization...")
    # print("Max value after reversing: {}".format(np.amax(data)))
    # print("Min value after reversing: {}".format(np.amin(data)))
    return data