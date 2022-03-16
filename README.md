# AAPSO
# An Adaptive and Altruistic PSO-based Deep Feature Selection Method for Pneumonia Detection from Chest X-Rays
**


## Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

## Running the codes:
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
Then, run the code using the command prompt as follows:

`python main.py --data_directory "data"`

Available arguments:
- `--epochs`: Number of epochs of training. Default = 20
- `--learning_rate`: Learning Rate. Default = 0.0001
- `--batch_size`: Batch Size. Default = 32

## Citation:
If this article helps in your research in any way, please cite us using:

```
@article{
}
```
