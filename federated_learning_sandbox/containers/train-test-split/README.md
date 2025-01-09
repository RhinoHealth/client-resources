# Generalized Compute Example - Train Test Split
<br/>

### **Description**

This example provides files that can be used with the Rhino Health Generalized Compute capability to remotely split an input cohort into two output cohorts on a Rhino Client.

It shows how to:
* Process an input CSV file as a dataframe
* Create multiple output CSV files from this input
* Use a multi-step Dockerfile to build the container image (using a separate step for installing requirements)

Please reference the User Documentation and/or Tutorials for in depth explanations on how to use the Generalized Compute capability.
<br/><br/>

### **Resources**
- `Dockerfile` - This is the Dockerfile to be used for building the container image
- `train_test_split.py` - This file contains the python code for splitting input cohort (using sklearn.model_selection.train_test_split)
- `run_code.sh` - The entrypoint shell script for the docker container, which runs train_test_split.py 
- `requirements.txt` - The python requirements for this project
<br><br>

# Getting Help
For additional support, please reach out to [support@rhinohealth.com](mailto:support@rhinohealth.com).
