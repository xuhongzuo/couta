# COUTA  - time series anomaly detection
Implementation of an unsupervised time series anomaly detection model based on calibrated one-class classification.  
The full paper, **"Calibrated One-class Classification for Unsupervised Time Series Anomaly
Detection"** is available at [link]().  
Please consider citing our paper if you find this repository useful. :wink:
```
bib files (TBD)
```

  
## Environment  
main packages
```  
torch==1.10.1+cu113  
numpy==1.20.3  
pandas==1.3.3  
scipy==1.4.1  
scikit-learn==1.1.1  
```  
we provide a `requirements.txt` in our repository.
  
  
  
## Takeaways
COUTA provides easy APIs in a sklearn style, that is, we can first instantiate the model class by giving the parameters
```python
from src.algorithms.couta_algo import COUTA
model_configs = {'sequence_length': 50, 'stride': 1}
model = COUTA(**model_configs)
```
then, the instantiated model can be used to fit and predict data, please use dataframes of pandas as input data
```python
model.fit(train_df)
score_dic = model.predict(test_df)
score = score_dic['score_t']
```
We use a dictionary as our prediction output for the sake of consistency with [an evaluation work of time series anomaly detection](https://github.com/astha-chem/mvts-ano-eval)  
`score_t` is a vector that indicates anomaly scores of each time observation in the testing dataframe, and a higher value represents a higher likehood to be an anomaly
  
  
  
## Datasets used in our paper
*Due to the license issue of these datasets, we provide download links here, but we offer the preprocessing scripts in `data_preprocessing.ipynb`, and you can easily generate processed datasets that can be directly fed into our pipeline by downloading original data and running this notebook. *  

The used datasets can be downloaded from:  
- ASD   https://github.com/zhhlee/InterFusion  
- SMD   https://github.com/NetManAIOps/OmniAnomaly  
- SWAT  https://itrust.sutd.edu.sg/itrust-labs_datasets  
- WaQ   https://www.spotseven.de/gecco/gecco-challenge  
- DSADS https://github.com/zhangyuxin621/AMSL  
- Epilepsy https://github.com/boschresearch/NeuTraL-AD/  
  
  
  
## Reproduction of experiment results
### Experiments of the effectivness (4.2)
After handling the used datasets, you can use `main.py` to perform COUTA on different time series datasets, we use six datasets in our paper, and `--data` can be chosen from `[ASD, SMD, SWaT, WaQ, Epilepsy, DSADS]`.

For example, perform COUTA on the ASD dataset by
```shell
python main.py --data ASD --algo COUTA
```
or you can directly use `script_effectivenss.sh`  

### Generalization test (4.3)
we include the used synthetic datasets in `data_processed/`
```shell
python main_showcase.py --type point
python main_showcase.py --type pattern
```
two anomaly score `npy` files are generated, you can use `experiment_generalization_ability.ipynb` to visualize the data and our results.

### Robustness (4.4)
use `src/experiments/data_contaminated_generator_dsads.py` and  `src/experiments/data_contaminated_generator_ep.py` to generate datasets with various contamination ratios  
use `main.py` to perform COUTA on these datasets, or directly execute `script_robustness.sh`

### Ablation study (4.5)
change the `--algo` argument to `COUTA_wto_umc`, `COUTA_wto_nac`, or `Canonical`, e.g., 
```shell
python main.py --algo COUTA_wto_umc --data ASD
```
use `script_effectiveness.sh` also produce detection results of ablated variants  

### Others
As for the sensitivity test (4.6), please adjust the parameters in the yaml file.  
As for the scalability test (4.7), the produced result files also contain execution time.  
  
  
  
## Competing methods
All of the anomaly detectors in our paper are implemented in Python. We list their publicly available implementations below. 
- `OCSVM` and `ECOD` :  we directly use pyod (python library of anomaly detection approaches); 
- `GOAD`: https://github.com/lironber/GOAD 
- `DSVDD`: https://github.com/lukasruff/Deep-SVDD-PyTorch 
- `USAD`: https://github.com/hoo2257/USAD-Anomaly-Detecting-Algorithm
- `GDN`: https://github.com/d-ailin/GDN
- `NeuTraL`: https://github.com/boschresearch/NeuTraL-AD
- `TranAD`: https://github.com/imperial-qore/TranAD
- `LSTM-ED`, `Tcn-ED`, `MSCRED` and `Omni`: https://github.com/astha-chem/mvts-ano-eval/
