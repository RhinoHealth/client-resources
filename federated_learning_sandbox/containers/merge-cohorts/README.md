# Generalized Compute Example - Merge Cohorts Data
<br/>

### **Description**

This example provides files that can be used with the Rhino Health Generalized Compute capability to remotely merge two input cohorts into a single output cohort on a Rhino Client.

It shows how to:
* Process multiple input CSV files
* Merge the inputs into a single output CSV file
* Use a single-step Dockerfile to build the container image (without using a separate step for installing requirements)

Please reference the User Documentation and/or Tutorials for in depth explanations on how to use the Generalized Compute capability.
<br/><br/>

### **Resources**
- `Dockerfile` - This is the Dockerfile to be used for building the container image
- `merge_cohorts_data.py` - This file contains the python code for merging the input cohorts
- `requirements.in` - The input python requirements for this project
- `requirements.txt` - The compiled python requirements for this project (using `pip-compile` on the requirements.in file)
<br><br>

# Getting Help
For additional support, please reach out to [support@rhinohealth.com](mailto:support@rhinohealth.com).
