import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os
import numpy as np
from scipy.signal import butter, filtfilt
import numpy as np
from sklearn.cluster import KMeans

AbsoluteDifference = []
PercentageDifference = []
TimeDeviation = []

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

itteration = 0
# moving average function
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

for i in os.listdir('./Intitial File/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'):
    print()
    print()
    print()
    print(i, itteration + 1)
    if i == 'xa.s12.00.mhz.1971-03-25HR00_evid00028.csv':
        next
    
    if os.path.splitext(i)[1] == '.csv':
    
        cat_directory = './Intitial File/space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
        cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
        data_cat = pd.read_csv(cat_file)

        test_filename = i
        data_directory = './Intitial File/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
        csv_file = f'{data_directory}/{test_filename}'
        data = pd.read_csv(csv_file)
        file_path = csv_file
        
        # Exclude datetime column
        # data_values = data.drop('timestamp',axis=1).values
        data_values = data.drop(data.columns[[0, 1]], axis=1).values

        if len(data_values) >= 572418:
            data_values = data_values[0:572418]

        # Convert data to float type
        data_values = data_values.astype('float32')
        # print('original shape - >' , data_values.shape)

        original_shape = data_values.shape

        data_values = data_values.reshape(-1)
        #moving average
        data_values = moving_average(data_values, window_size=100)

        # Filter your data
        data_values = bandpass_filter(data_values, lowcut=0.5, highcut=1.0, fs=7)

        # Normalizing data to range [-1, 1]
        data_values = 2 * ((data_values - data_values.min()) / (data_values.max() - data_values.min())) - 1

        # print(data_values)
        # print(max(data_values))
        # print(min(data_values))

        # reshape array
        data_values = data_values.reshape(original_shape)
        # print('reshaped array - >' , data_values.shape)

        # Create new dataframe with converted values
        data_converted = pd.DataFrame(data_values, columns=['velocity'])

        # Add back datetime column
        # data_converted.insert(0, 'timestamp', data['timestamp'])

        data_converted.insert(0, 'timestamp', data['time_rel(sec)'])


        # Extract the filename from the file path
        file_name = os.path.basename(file_path)

        # get real impac time
        # Path to the catalog file
        catalog_file_path = cat_file

        # Read the catalog CSV file
        catalog = pd.read_csv(catalog_file_path)

        # Extract the relative time for the corresponding filename

        file_name, _ = os.path.splitext(file_name)

        relative_time = catalog.loc[catalog['filename'] == file_name, 'time_rel(sec)']

        print(relative_time)
        print(relative_time.values[0])
        # Check if the filename exists in the catalog and output the result
        if not relative_time.empty:
            print(f'Relative time for {file_name} : {relative_time.values[0]} seconds')
        else:
            print(f'Filename {file_name} not found in the catalog.')
            next

        data_converted = data_converted.dropna()
        
        data_tensor = tf.convert_to_tensor(data_converted.drop('timestamp', axis=1).values, dtype=tf.float32)

        # Load the saved model from the specified path
        autoencoder = load_model('./Intitial File/autoencoder_model.h5',
                                custom_objects={'mse': MeanSquaredError})
        autoencoder.compile(optimizer='adam', loss='mse')

        # Calculate the reconstruction error for each data point
        reconstructions = autoencoder.predict(data_tensor)
        mse = tf.reduce_mean(tf.square(data_tensor - reconstructions),axis=1)
        anomaly_scores = pd.Series(mse.numpy(), name='anomaly_scores')
        anomaly_scores.index = data_converted.index

        threshold = anomaly_scores.quantile(0.99)
        anomalous = anomaly_scores > threshold
        binary_labels = anomalous.astype(int)
        precision, recall, f1_score, _ = precision_recall_fscore_support(binary_labels, anomalous, average='binary')


        print(len(anomalous))
        true_indexes = [i for i, x in enumerate(anomalous) if x]
        # print("Indexes of True values:", true_indexes)

        # print("lenght of True values:", len(true_indexes))


        # Iterating over the DataFrame using the specified indices
        anamolic_time_stamps = []
        for index in true_indexes:
            row = data_converted.loc[index]
            anamolic_time_stamps.append(row['timestamp'])
            # print(f"Index: {index}, Row: {row.to_dict()}")

        # the average of the timestamps will be the predicted answer.
        # model predicts the index of those timestaps then we fetch the only predicted timestamps from the csv and take average.
        import numpy as np
        from sklearn.cluster import KMeans


        # Extract only the anomalous timestamps and velocities for clustering
        anomalous_data = data_converted.loc[true_indexes, ['timestamp', 'velocity']].values

        anomalous_timestamps = data_converted.loc[true_indexes, 'timestamp'].values

        # Sort the timestamps to ensure they are in order
        sorted_anomalous_timestamps = np.sort(anomalous_timestamps)

        # Initialize variables for dynamic clustering
        groups = []
        current_group = []

        # Iterate through sorted timestamps and group based on proximity
        for i, timestamp in enumerate(sorted_anomalous_timestamps):
            if len(current_group) == 0:
                current_group.append(timestamp)
            else:
                # Check if the current timestamp is within 10 relative seconds of the last one
                if timestamp - current_group[-1] <= 20:
                    current_group.append(timestamp)
                else:
                    # Check if the group has at least 25 anomalies before moving to the next group
                    if len(current_group) >= 1:
                        groups.append(current_group)
                    # Start a new group
                    current_group = [timestamp]

        # Handle the last group if it has at least 25 anomalies
        if len(current_group) >= 100:
            groups.append(current_group)


        avg_impact_time = 0
        clusters_amount = []
        # Display results for each group
        for i, group in enumerate(groups):
            avg_impact_time = np.mean(group)
            clusters_amount.append([len(group), group])
            # print(f"Group {i+1}: Number of points = {len(group)}, Avg. Impact Time = {avg_impact_time}")


        max_value = max(clusters_amount, key=lambda x: x[0])
        avg = (anamolic_time_stamps[0])
        avg = max_value[1][0]



        # Absolute Difference
        difference = abs(avg - relative_time.values[0])
        AbsoluteDifference.append(difference)
        print(f"Absolute Difference: {difference}")

        # Percentage Difference
        percentage_difference = (difference / relative_time.values[0]) * 100
        PercentageDifference.append(percentage_difference)
        print(f"Percentage Difference: {percentage_difference:.2f}%")
        
        Time_Deviation = (abs(avg - relative_time.values[0])/relative_time.values[0]) *100
        TimeDeviation.append(Time_Deviation)
        print(Time_Deviation)


        # Convert the boolean list to a TensorFlow tensor
        anomalous_tensor = tf.constant(anomalous)

        # 1. Get the values from `data_tensor` that correspond to `True` in `anomalous`
        true_values = tf.boolean_mask(data_tensor, anomalous_tensor)
        # print("Values corresponding to True indices in data_tensor:", true_values.numpy())

        # 2. Get the indexes of the `True` values in the `anomalous` list
        true_indexes = tf.where(anomalous_tensor).numpy().flatten()
        # print("Indexes of True values:", true_indexes)

        # Optional: Print the values along with their corresponding indexes
        # for index in true_indexes:
        #     print(f"Index: {index}, Value: {data_tensor[index].numpy()}")

        # test = data_converted['value'].values
        test = data_converted['velocity'].values

        predictions = anomaly_scores.values

        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1 Score: ", f1_score)

        import matplotlib.pyplot as plt

        # Plot the data with anomalies marked in red
        plt.figure(figsize=(16, 8))
        plt.plot(data_converted['timestamp'], data_converted['velocity'], label='Velocity')

        # Mark anomalies in red on the same plot
        plt.plot(data_converted['timestamp'][anomalous], data_converted['velocity'][anomalous], 'ro', label='Anomalies')

        # Draw a green vertical line at the exact index of `relative_time.values[0]`
        plt.axvline(x=relative_time.values[0], color='black', linestyle='--', linewidth=3 , label='Rel. Arrival')

        plt.axvline(x=avg, color='green', linestyle='--', linewidth=3, label='Avg. Predicted Rel. Arrival')

        # Add title and labels
        plt.title(f'Anomaly Detection, Filename : {file_name}')
        plt.xlabel('Time')
        plt.ylabel('Velocity')

        # Show the legend
        plt.legend()
        
        # Directory path
        directory_path = './Intitial File/SavedPNGs/Lunar/training/'
        os.makedirs(directory_path, exist_ok=True)  # Create the directory if it doesn't exist

        # Save the figure
        plt.savefig(f'{directory_path}/{i}.png', format='png')  # Make sure to add .png extension




        print(i)
    else:
        print('skip')
        
        
df = pd.DataFrame(
    {
        'AbsoluteDifference': AbsoluteDifference,
        'PercentageDifference': PercentageDifference,
        'TimeDeviation': TimeDeviation
    }
)

df.to_csv('output.csv', index=False)
