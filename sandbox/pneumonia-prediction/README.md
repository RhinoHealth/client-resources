## Conducting an end-to-end ML project using multimodal data from multiple hospitals

![Model Sandbox Overview](./img/sandbox_overview.jpg)

**Follow the Tutorial: <a target="_blank" href="https://docs.rhinohealth.com/hc/en-us/articles/15586509051549-Pneumonia-Prediction-Step-1-Scenario-FCP-Overview">Here</a>**

## Notebook 1: Import EHR and CXR Datasets
In this step, we gain experience in importing multimodal datasets onto the Rhino FCP. We will use MIMIC-IV as the source to create datasets from a database that hosts EHR data. In addition, we will load CXR DICOM images, which are linked to the EHR tables using a mapping lookup table.

## Notebook 2: Data Engineering 
 In this step, the the DICOM files (chest x-rays) are converted to JPEG.

## Notebook 3: Exploratory data analysis & biostatistical methods
In this step, the user will gain experience using privacy preserving exploratory data analytics capabilities of Rhino. We show few privacy preserving query with metrics.


## Notebook 4: Prepare Models for FL Training and execute model training
In this step, you will split data into training and testing and use the datasets for model training. You will gain experience converting an existing model to NVFlare and execute the FL model training across two sites. 

## Notebook 5: Federated Evaluations
Visualize evaluation results


Need Help?
<a target="_blank" href="https://docs.rhinohealth.com/">Rhino Health Documenation Center</a> or <a target="_blank" href="mailto:support@rhinohealth.com">support@rhinohealth.com</a>