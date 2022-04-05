## Quick Start

For running the experiment, please refer to [the first chapter](#instructions-on-how-to-run-the-code). For the OLS analysis, you can directly jump to [the second chapter](#instructions-on-how-to-recover-the-ols-tables).

### Instructions on how to run the code

For this part, use the code under the "TransferAttackIRL" directory.
#### Download datasets & pretrained models
- Download the pretrained models to "./SurrogateModel/PretrainedModels"
- Download the Adience and ImageNet dataset to "./SurrogateDataset".

#### Preprocess datasets
Preprocess the datasets using scripts in "./RawDatasetPreprocess":
 - Use preprocess_adience.py to preprocess the Adience dataset.
 - Use generateAEcandidates_adience.py to generate AE (Adversarial Example) candidates stored in the .npy file.

#### Train local surrogate models
 - Train local surrogate models (raw & augmented) using scripts in "./SurrogateModel"
- Use command "bash run_train.sh" and set parameters following the instructions.
 - Train local surrogate models (adversarial) using scripts in "./Defenses"
- Use command "bash run_train_adversarial.sh" and set parameters following the instructions.

#### Generate AEs
Run adversarial attack methods and store the generated AEs using scripts in ".\Attacks". The generated AEs will be saved to ".\AdversarialExample".
- To generate AEs corresponding to the 9 adversarial attack methods, use command "bash run_attacks.sh" and set parameters following the instructions.
- To generate AEs corresponding to the CW attack with different settings of kappa, use command "bash run_attacks_cw2_kappa.sh" and set parameters following the instructions.

#### Decompress the AEs
Decompress the xx.npy files to AE images. The generated AE images will be saved to ".\AdversarialExample".
- To decompress AEs corresponding to the 9 adversarial attack methods, use command "bash generate_imgs.sh" and set parameters following the instructions.
- To decompress AEs corresponding to the CW attack with different settings of kappa, use command "bash generate_imgs_kappa.sh" and set parameters following the instructions.

#### Test AEs against cloud APIs
Test the generated AEs against APIs provided by platforms such as AWS, Aliyun and Google Vision.
 - Firstly, you need to manually set configurations for the cloud APIs, including access_key, api_key and so on. Please check xx_tester.py files at "./Cloud/Tester".
 - Then, use the script at "./Cloud/test_ae.py" and assign the dataset/pretrain/platform parameters. The test results will be saved to ".\Cloud\Results".

#### Analyze cloud test results
Analyze cloud test results using the script "./Cloud/analyze.py".

### Instructions on how to recover the OLS tables
For this part, use the code and data under the "results" directory. You may use this part independently to the previous chapter because we have included our data.

#### Preparing the cloud results
You may use the descriptions in [the first chapter](#instructions-on-how-to-run-the-code) to get the cloud results. We have included ours in the "./cloud_results/" folder so that you are not required to run your own experiments.

#### Run the OLS analysis
We provide two OLS notebooks, respectively, for the ImageNet dataset and the Adience dataset. These notebooks can be run directly. Both notebooks are documented by comments.
