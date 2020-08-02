
import pandas as pd
from ../model_inference import ModelInference
from ../model_training import ModelTraining
from ../experiments import Experiments


df = pd.read_csv('../data/train.csv')

Experiments().run_experiment()

model = ModelTraining().model_training()

pred = ModelInference().predict()