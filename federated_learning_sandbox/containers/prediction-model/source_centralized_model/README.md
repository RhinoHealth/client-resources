**How to Transform a Centralized Training Model to NVFLARE**

* Begin with a centralized model, for example: https://www.kaggle.com/code/fahadmehfoooz/pneumonia-classification-using-pytorch/notebook
* Follow these steps:
   1. Custom folder:
     Create your network.py, trainer.py, and copy relevant functions to your trainer.py 
   2. Config folder:
     Create config_fed_client.json and config_fed_server.json 
   3. infer.py (optional)
     Create your model validation code (to be run using Generalized Compute)
   4. Try everything out with the docker_run.sh utility
   5. Push your container to your ECR repo with the docker_push.sh utility
   6. Run training and validation via the Rhino Health Platform in the UI or SDK
* Example: Pneumonia Classification Model:
   * Source: source_centralized_model/pneumonia_classification.py
   * Converted to: custom/[network.py + pneumonia_trainer.py (+pt_constants.py)] + infer.py 

  **Training**
    
  |Step| Source | NVFLARE| 
  |---|:---:|---:|
   |data transform|lines 9-30|pneumonia_trainer.py: lines 66-73|
    |load data|lines 33-52|pneumonia_trainer.py: lines 75-82|
    |define model|lines 54-85|network.py|
    |model params|lines 91-99|pneumonia_trainer.py: lines 59-64|
    |training|lines 101-128|pneumonia_trainer.py: lines 90-116|
    |NVFLARE wrapper|___|pneumonia_trainer.py: lines 118-186|
    
    **Inference**
    
  |Step| Source | NVFLARE| 
  |---|:---:|---:|
   |data transform|lines 9-30|infer.py: lines 24-30|
    |load data|lines 33-52|infer.py: lines 31-33|
    |define model|lines 54-85|network.py|
    |infer|lines 130-146|infer.py: lines 35-43|
    |write cohort|___|infer.py: lines 47-55|
  
