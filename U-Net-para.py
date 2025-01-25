import os
import tensorflow as tf
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set environment variables for parallelism (40 cores)
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['TF_NUM_INTRAOP_THREADS'] = '40'
os.environ['TF_NUM_INTEROP_THREADS'] = '40'

# Limit TensorFlow to use 40 cores for parallel computation
tf.config.threading.set_intra_op_parallelism_threads(40)
tf.config.threading.set_inter_op_parallelism_threads(40)

# Function to replace NaN values with the mean of each variable
def replace_nan_with_mean(data):
    nan_mean = np.nanmean(data)
    data[np.isnan(data)] = nan_mean
    return data

# Load the dataset
netcdf_file = r"/scratch/20cl91p02/ANN_BIO/Unet/ann_input_data.nc"
ds = xr.open_dataset(netcdf_file)

# Extract data variables
fe = ds['fe'].values
po4 = ds['po4'].values
si = ds['si'].values
no3 = ds['no3'].values  # Predictor
nppv = ds['nppv'].values  # Target variable

# Extract latitude and longitude
latitude = ds['latitude'].values
longitude = ds['longitude'].values

# Since depth is constant, discard the depth dimension and focus on (time, lat, lon)
fe = fe[:, 0, :, :]
po4 = po4[:, 0, :, :]
si = si[:, 0, :, :]
no3 = no3[:, 0, :, :]
nppv = nppv[:, 0, :, :]

# Replace NaN values in predictors and target using replace_nan_with_mean
fe = replace_nan_with_mean(fe)
po4 = replace_nan_with_mean(po4)
si = replace_nan_with_mean(si)
no3 = replace_nan_with_mean(no3)
nppv = replace_nan_with_mean(nppv)

# Stack the input variables along a new channel dimension (fe, po4, si, no3)
inputs = np.stack([fe, po4, si, no3], axis=-1)

# Prepare input for LSTM
time_steps = 5
samples = inputs.shape[0] - time_steps
X_lstm = np.array([inputs[i:i + time_steps] for i in range(samples)])
y_lstm = nppv[time_steps:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[2] * X_train.shape[3] * X_train.shape[4])
X_test_reshaped = X_test.reshape(-1, X_test.shape[2] * X_test.shape[3] * X_test.shape[4])
X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

scaler_y = StandardScaler()
y_train_reshaped = y_train.reshape(-1, y_train.shape[1] * y_train.shape[2])
y_test_reshaped = y_test.reshape(-1, y_test.shape[1] * y_test.shape[2])
y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)

# Define the UNetBlock class
class UNetBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(UNetBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.up2 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.up1 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv10 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')

    def call(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.conv2(conv1)
        pool1 = self.pool1(conv1)
        conv2 = self.conv3(pool1)
        conv2 = self.conv4(conv2)
        pool2 = self.pool2(conv2)
        conv3 = self.conv5(pool2)
        conv3 = self.conv6(conv3)
        up2 = self.up2(conv3)
        up2 = tf.image.resize_with_crop_or_pad(up2, conv2.shape[1], conv2.shape[2])
        up2 = tf.keras.layers.concatenate([conv2, up2], axis=-1)
        conv4 = self.conv7(up2)
        conv4 = self.conv8(conv4)
        up1 = self.up1(conv4)
        up1 = tf.image.resize_with_crop_or_pad(up1, conv1.shape[1], conv1.shape[2])
        up1 = tf.keras.layers.concatenate([conv1, up1], axis=-1)
        conv5 = self.conv9(up1)
        conv5 = self.conv10(conv5)
        return conv5

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape
        return (batch_size, height, width, 64)

# Define the U-Net + LSTM Model
inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
x = tf.keras.layers.TimeDistributed(UNetBlock())(inputs)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((-1,)))(x)
x = tf.keras.layers.LSTM(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(y_train.shape[1] * y_train.shape[2])(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled.reshape(y_train_scaled.shape[0], -1), 
                    epochs=50, batch_size=16, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test_scaled.reshape(y_test_scaled.shape[0], -1))

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_y = scaler_y.inverse_transform(predictions.reshape(-1, y_test.shape[1] * y_test.shape[2])).reshape(y_test.shape)

# Replace NaN values in the predicted output and actual values
predicted_y = replace_nan_with_mean(predicted_y)
average_actual_nppv = replace_nan_with_mean(nppv[time_steps:])

# Define output file path
output_file_path = r"/scratch/20cl91p02/ANN_BIO/Unet/average_output_unet+lstm_nppv.nc"

# Create a new NetCDF file
with xr.Dataset() as ds_out:
    ds_out.coords['latitude'] = ('latitude', latitude)
    ds_out.coords['longitude'] = ('longitude', longitude)
    ds_out['predicted_nppv'] = (('latitude', 'longitude'), predicted_y)
    ds_out['actual_nppv'] = (('latitude', 'longitude'), average_actual_nppv)
    ds_out.to_netcdf(output_file_path)

print("Test Loss:", test_loss)
