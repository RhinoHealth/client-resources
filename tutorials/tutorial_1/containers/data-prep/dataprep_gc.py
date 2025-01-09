import pandas as pd
import os
import pydicom
import numpy as np
from PIL import Image
from sklearn.impute import SimpleImputer
import glob


def convert_dcm_image_to_jpg(name):
	dcm = pydicom.dcmread(name)
	img = dcm.pixel_array.astype(float)
	rescaled_image = (np.maximum(img, 0) / img.max()) * 255  # float pixels
	final_image = np.uint8(rescaled_image)  # integers pixels
	final_image = Image.fromarray(final_image)
	return final_image


def dataset_dcm_to_jpg(dataset_df):
	input_dir = '/input/dicom_data/'
	output_dir = '/output/0/file_data/'
	dcm_list = glob.glob(input_dir + '/*/*.dcm')

	for dcm_file in dcm_list:
		image = convert_dcm_image_to_jpg(dcm_file)
		jpg_file_name = dcm_file.split('/')[-1].split('.dcm')[0] + '.jpg'
		ds = pydicom.dcmread(dcm_file)
		idx = dataset_df['Pneumonia'][dataset_df.SeriesUID == ds.SeriesInstanceUID].index[0]
		ground_truth = '1' if dataset_df.loc[idx, 'Pneumonia'] else '0'
		class_folder = output_dir + ground_truth
		if not os.path.exists(class_folder):
			os.makedirs(class_folder)
		image.save('/'.join([class_folder, jpg_file_name]))
		dataset_df.loc[idx, 'JPG file'] = '/'.join([ground_truth, jpg_file_name])

	return dataset_df


if __name__ == '__main__':
	# Read dataset from /input
	dataset = pd.read_csv('/input/dataset.csv')

	# Run data imputation
	W_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	H_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	dataset['Weight'] = W_imp.fit_transform(dataset.Weight.values.reshape(-1, 1))
	dataset['Height'] = H_imp.fit_transform(dataset.Height.values.reshape(-1, 1))
	dataset['BMI'] = dataset.Weight / (dataset.Height ** 2)

	# Convert DICOM to JPG
	dataset = dataset_dcm_to_jpg(dataset)

	# Write dataset to /output
	dataset.to_csv('/output/0/dataset.csv', index=False)

