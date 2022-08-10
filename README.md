# AAPSO [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-adaptive-and-altruistic-pso-based-deep/classification-on-chest-x-ray-images)](https://paperswithcode.com/sota/classification-on-chest-x-ray-images?p=an-adaptive-and-altruistic-pso-based-deep)
# Deep Feature Selection for Pneumonia Detection
"An Adaptive and Altruistic PSO-based Deep Feature Selection Method for Pneumonia Detection from Chest X-Rays" accepted for publication in Applied Soft Computing, Elsevier
```
@article{
}
```
# Dataset Links
1. [Paul Monney's Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. [UCI datasets: In this repository](https://github.com/rishavpramanik/AAPSO/tree/main/data/UCIcsv)
3. [Microarray Datasets: In this repository](https://github.com/rishavpramanik/AAPSO/tree/main/data/MicroarrayCsv)
4. [COVID-19 prediction: In this repository](https://github.com/rishavpramanik/AAPSO/tree/main/data/COVIDcsv)

# Instructions to run the code:

Required directory structure:

(Note: ``train`` and ``val`` contains subfolders representing classes in the dataset.)

```

+-- data
|   +-- .
|   +-- train
|   +-- val
+-- AAPSO.py
+-- main.py

```
1. Download the repository and install the required packages:
```
pip3 install -r requirements.txt
```
2. The main file is sufficient to run the experiments.
Then, run the code using linux terminal as follows:

```
python3 main.py --data_directory "data"
```

Available arguments:
- `--epochs`: Number of epochs of training. Default = 10
- `--learning_rate`: Learning Rate. Default = 0.0001
- `--batch_size`: Batch Size. Default = 32

3. Please don't forget to edit the above parameters before you start
