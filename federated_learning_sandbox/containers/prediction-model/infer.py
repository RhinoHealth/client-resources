#!/usr/bin/env python
import sys

import pandas as pd
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose,  Resize, RandomRotation, CenterCrop

from network import PneumoniaModel


def infer(model_weights_file_path):
    # Setup the model
    model = PneumoniaModel()
    model.load_state_dict(torch.load(model_weights_file_path)["model"])
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Preparing the dataset for testing.
    transforms = Compose([
        Resize(size=(256, 256)),
        RandomRotation(degrees=(-20, +20)),
        CenterCrop(size=224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tabular_data = pd.read_csv("/input/cohort_data.csv")
    dataset = torchvision.datasets.ImageFolder(root="/input/file_data", transform=transforms)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Inference: Apply model and add scores column.
    scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            output = model(images)
            batch_scores = torch.select(output, 1, 1)
            scores.extend([score.item() for score in batch_scores])
    tabular_data['Model_Score'] = scores

    tabular_data.to_csv("/output/cohort_data.csv", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    (model_weights_file_path,) = args
    infer(model_weights_file_path)
    sys.exit(0)
