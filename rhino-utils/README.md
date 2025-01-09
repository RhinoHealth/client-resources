# Rhino Health Utilities

***Tools to assist in the local development, testing, and deployment to the FCP for both Generalized Compute and NVFLARE-based containers.***

<br><br>

# Why Use This?

* Iterate quickly while developing locally.
* Emulate the conditions of the Rhino Health FCP, to help ensure your container images will work there.
* These tools take care of many small details for you.  Some are crucial, such as building container images for the correct architecture (amd64, not ARM).  Some make local development faster and nicer, such as re-using local caches for `pip install` and `apt-get install`, and not running containers as a root user.
* `docker-run.sh --auto` completely automates setting up a local NVFLARE server and local client(s) and running FL training with them, saving you the time and effort of doing so repeatedly (and of learning how!).

<br><br>

# Requirements

1. Bash shell available at `/bin/bash` (the common default location)
2. [Docker](https://docs.docker.com/get-docker/)

<br><br>

# Included Tools

Run each tool with `-h` or `--help` for usage details.

1. `docker-push.sh`: Build a docker container image and push it to a container image repository. For more usage detail, see [Using the Docker Push Script](#using-the-docker-push-script).
2. `drive_admin_api.py`: A utility Python script used to help facilitate the NVFlare FL network when using the `nvflare-docker-run.sh` script
3. `gc-docker-run.sh`: Build a docker container image and run it locally. For more usage detail, see [Using the Generalized Compute Run Script](#using-the-generalized-compute-docker-run-script).
4. `nvflare-docker-run.sh`: Run FL training with your code using, locally, using Docker.  Recommended: `--auto`. For more usage detail, see [Using the NVFlare Docker Run Script](#using-the-nvflare-docker-run-script).
5. `nvflare-docker-run-inference.sh`: Run inference with your code on a single dataset. For more usage detail, see [Using the Docker Push Script](#using-the-docker-push-script).
6. `run_inference.sh`: A utility Shell script used to help facilitate the `nvflare-docker-run-inference.sh` script
7. `upload-file-to-s3.sh`: A utility Shell script to make files available to FCP at run time, so they can be used at when running your code object without having to re-create it, making your code-object containers smaller, faster, and re-usable. See [this example](examples/rhino-sdk/runtime-external_files.ipynb) for how to use this utility in context.

<br><hr><br>

## Input Data Directory Structure

```
| /input/<br>
|_____ 0/ - Representing the first Dataset selected
|___________ dataset.csv - The tabular data of the given dataset
|___________ dicom_data/ - If applicable, associated DICOM data
|___________ file_data/ - If applicable, associated file data
|_____ 1/ - Representing the second Dataset selected
|___________ dataset.csv - The tabular data of the given dataset
|___________ dicom_data/ - If applicable, associated DICOM data
|___________ file_data/ - If applicable, associated file data
...
```
**Note**: The same structure applies for the `/output` folder. If you are interested in saving data you must format the data in this structure.

<br><hr><br>

## Using the Generalized Compute Docker Run Script

1. Adapt your code to read all inputs from `/input` and write all results to `/output`.
   See [Input Data Directory Structure](#input-data-directory-structure) for more details.
2. List the Python dependencies of your code in a `requirements.txt` file. (**Recommended**: Use "pinned dependencies" where possible.)
3. Copy `client-resources/tutorials/tutorial_1/containers/data-prep/Dockerfile` and make changes as need where there are comments beginning with: "`# !! EDIT THIS:`"
4. Test by building and running the container locally: (**Note:** The first two lines below only need to be run once) 
```shell
cd path/to/client-resources/rhino-utils
chmod +x *.sh
cd path/to/your/code
./path/to/client-resources/rhino-utils/gc-docker-run.sh path/to/input path/to/output
```
5. Debug and repeat until it works...

<br><hr><br>

## Using the NVFlare Docker Run Script

1. Adapt your FL client code to read all inputs from `/input`. See [Input Data Directory Structure](#input-data-directory-structure) for more details.
2. Adapt your FL server code to write output model parameters (weight, biases etc.) to a file named `model_parameters.*` under `/output`.  (e.g. `/output/model_parameters.pt`)
3. List the Python dependencies of your code in a `requirements.txt` file. (**Recommended**: Use "pinned dependencies" where possible.)
4. Copy `client-resources/tutorials/tutorial_1/containers/prediction-model/Dockerfile` and make changes as needed where there are comments beginning with: "`# !! EDIT THIS:`"
5. Test by building and running the container locally: (**Note:** The first two lines below only need to be run once) 
```shell
cd path/to/client-resources/rhino-utils
chmod +x *.sh 
cd path/to/your/code
./path/to/client-resources/rhino-utils/nvflare-docker-run.sh --auto path/to/input path/to/output
```
6. Debug and repeat until it works...

<br><hr><br>
## Using the NVFlare Docker Run Inference Script

1. Complete all steps in [Using the NVFlare Docker Run Script](#using-the-nvflare-docker-run-script) above.
5. Run Inference by building and running the container locally: (**Note:** The first two lines below only need to be run once) 
```shell
cd path/to/client-resources/rhino-utils
chmod +x *.sh 
cd path/to/your/code
./path/to/client-resources/rhino-utils/nvflare-docker-run-inference.sh path/to/input path/to/output path/to/weights
```

<br><hr><br>

## Using the Docker Push Script

1. If you have not already done so, authenticate to your cloud provider CLI. For AWS, set up your ECR credentials with the AWS CLI. If you need assistance in setting these up, please follow the instructions in the `Setup` steps of [Pushing Generalized Compute Image Containers](https://docs.rhinohealth.com/hc/en-us/articles/6040656682269-Pushing-Generalized-Compute-Container-Images) guide. If your credentials have been updated and you need assistance updating them in your local config, please review this guide: [How can I setup or Change my ECR Credentials?](https://rhinohealth.zendesk.com/hc/en-us/articles/11383336127133). For GCP, run `gcloud auth login` and follow the instructions.
2. Find your workgroup repository name in the container image registry (e.g. `rhino-gc-workgroup-my-workgroup`). If you need help finding your workgroup repository name, please follow this guide [How can I find my Workgroup Repository Name And My Image Registry Credentials?](https://docs.rhinohealth.com/hc/en-us/articles/12780382529309-How-can-I-find-my-SFTP-Server-Name-IP-Address-SFTP-Username-SFTP-Password-ECR-Workgroup-Repository-Name-ECR-Access-Key-ID-ECR-Secret-Access-Key)
3. Choose a unique tag for this container image (e.g. `tutorial-1-data-prep-v1`)
4. Find the rhino domain you are using. For AWS based environments, this is `rhinohealth.com`. For GCP based environments, this is `rhinofcp.com`. The default value is `rhinohealth.com`. If you wish to set a different default value, you can do so by setting the `RHINO_DOMAIN` environment variable. 
5. Build and push the container to the Rhino ECR repo using `docker-push.sh`.  <br>For example:
```shell
cd path/to/client-resources/rhino-utils
chmod +x *.sh
cd path/to/your/code
./path/to/client-resources/rhino-utils/docker-push.sh rhino-gc-workgroup-my-workgroup tutorial-1-data-prep-v1
```
<br><br>

# Getting Help
For additional support, please reach out to [support@rhinohealth.com](mailto:support@rhinohealth.com).
