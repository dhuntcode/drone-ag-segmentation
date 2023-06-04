import yaml
from pydantic import BaseModel, ValidationError


class TrainConfig(BaseModel):
    image_dir: str
    mask_dir: str
    num_epochs: int
    learning_rate: float
    batch_size: int


class TestConfig(BaseModel):
    image_dir: str
    mask_dir: str
    num_classes: int
    batch_size: int
    checkpoint_path: str


def load_yaml(path, mode):
    """
    Load YAML configuration file and validate the contents based on the specified mode.

    Args:
        path (str): Path to the YAML configuration file.
        mode (str): Mode indicating the type of configuration to validate.

    Returns:
        Union[TrainConfig, TestConfig]: Validated configuration object based on the specified mode.
    """
    
    with open(path, "r") as f:
        config_data = yaml.safe_load(f)

    if mode == "train":
        try:
            config = TrainConfig(**config_data["train"])
        except ValidationError as e:
            print(f"Error validating the configuration data: {e}")

    elif mode == "test":
        config = TestConfig(**config_data["test"])

    else:
        print(f"Error validating the configuration data")

    return config
