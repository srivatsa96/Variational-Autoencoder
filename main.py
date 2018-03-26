from configuration.config import ModelConfig, TrainConfig
from model.VariationalAutoencoder import VariationalAutoencoder
from operations.train import TrainEncoderModel



train_config = TrainConfig()
model_config = ModelConfig()
model = VariationalAutoencoder(config=model_config)
train = TrainEncoderModel(config=train_config,model=model)
train.train_model(restore_model=True)
