import os
import numpy as np
import matplotlib.pyplot as plt

def load_time_series_from_folder(dataset_path):
    time_series_data = []
    labels = []
    count_less100 = 0
    for foldername in os.listdir(dataset_path):
        folderpath = os.path.join(dataset_path, foldername)
        for filename in os.listdir(folderpath):
            ##if filename.endswith(".bin"):
            file_path = os.path.join(folderpath, filename)
            # Assuming the .bin files contain float32 data
            data = np.fromfile(file_path, dtype=np.float32)
            time_series_data.append(np.array(data))
            labels.append(foldername)

            if(len(data) != 100):
                count_less100 += 1
                print(filename)
    print(len(labels))
    print(count_less100)
    return time_series_data, labels

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    folder_path = os.path.join(os.path.dirname(script_path), "../DataSet_processed/PHASE_v4/PHASE_v4")
    time_series, labels = load_time_series_from_folder(folder_path)

    print("Number of time series data:", len(time_series))
    #if len(time_series) > 0:
        # print("Length of a single time series data:", time_series.shape)

    # Save time series data and labels
    np.save(os.path.join(os.path.dirname(script_path), 'data/v4_Phase_data.npy'), time_series)
    np.save(os.path.join(os.path.dirname(script_path), 'data/v4_labels.npy'), labels)