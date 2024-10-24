from functools import partial
from vla_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from vla_recipes.datasets.custom_dataset import get_custom_dataset, get_data_collator

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "custom_dataset": get_custom_dataset,
}
DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator
}