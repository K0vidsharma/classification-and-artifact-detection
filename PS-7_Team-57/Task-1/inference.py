import os
import json
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
from resnet import ResNet50


def make_prediction(folder_dir, filename, model, submission_lis):
    img_pth = os.path.join(folder_dir, filename)
    idx = int(filename.replace(".png", ""))
    img = Image.open(img_pth)
    img_tensor = ToTensor()(img).unsqueeze(0)
    logits = model(img_tensor)
    cls = torch.round(torch.sigmoid(logits))
    final_dict = {
        "index": idx,
        "prediction": "real" if cls == 1 else "fake"
    }
    submission_lis.append(final_dict)


def main():
    model = ResNet50(3, 1)
    checkpoint = torch.load('model_checkpoint.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    folder_dir = "perturbed_images_32"
    img_lis = [i for i in os.listdir(folder_dir) if i[-3:] == 'png']
    print(img_lis)
    submission_lis = []
    with torch.inference_mode():
        model.eval()
        for img in img_lis:
            make_prediction(folder_dir, img, model, submission_lis)
            print("predicted")


    print("Done")
    with open("submission.json", "w") as f:
        json.dump(submission_lis, f)
    print(submission_lis)


if __name__ == "__main__":
    main()





