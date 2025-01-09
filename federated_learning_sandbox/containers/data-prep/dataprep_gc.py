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


def cohort_dcm_to_jpg(df_cohort):
	input_dir = '/input/dicom_data/'
	output_dir = '/output/file_data/'
	dcm_list = glob.glob(input_dir + '/*/*.dcm')

	df_cohort['JPG_file'] = 'Nan'
	for dcm_file in dcm_list:
		image = convert_dcm_image_to_jpg(dcm_file)
		jpg_file_name = dcm_file.split('/')[-1].split('.dcm')[0] + '.jpg'
		ds = pydicom.dcmread(dcm_file)
		idx = df_cohort['Pneumonia'][df_cohort.SeriesUID == ds.SeriesInstanceUID].index[0]
		ground_truth = '1' if df_cohort.loc[idx, 'Pneumonia'] else '0'
		class_folder = output_dir + ground_truth
		if not os.path.exists(class_folder):
			os.makedirs(class_folder)
		image.save('/'.join([class_folder, jpg_file_name]))
		df_cohort.loc[idx, 'JPG file'] = '/'.join([ground_truth, jpg_file_name])

	return df_cohort


if __name__ == '__main__':
	# Read cohort from /input
	df_cohort = pd.read_csv('/input/cohort_data.csv')

	# Convert DICOM to JPG
	df_cohort = cohort_dcm_to_jpg(df_cohort)

	# Write cohort to /output
	df_cohort.to_csv('/output/cohort_data.csv', index=False)

