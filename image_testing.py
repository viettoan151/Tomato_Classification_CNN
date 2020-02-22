from main import Net
from main import model_name
from feature_extraction import classes
from feature_extraction import his_extract
from grabcut import grabcut_genmask
import argparse
import torch
import cv2
import numpy as np

def evaluate(image, model, device):
    img_X = int(200)
    img_Y = int(image.shape[0] * (img_X / image.shape[1]))
    imgresize = cv2.resize(image, (img_X, img_Y))

    # Apply Gaussian to remove noise
    image_blr = cv2.blur(imgresize, (5, 5))
    # Change to HSV color and extract histogram feature
    image_hsv = cv2.cvtColor(image_blr, cv2.COLOR_BGR2HSV)

    mask = grabcut_genmask(image_hsv, 6)
    imgcut = image_hsv * mask[:, :, np.newaxis]
    data = his_extract(imgcut)
    data = np.reshape(data,(1, 1, len(data)))
    tdata = torch.from_numpy(data).to(device, dtype=torch.float)
    model.eval()
    with torch.no_grad():
        output = model(tdata)
        pred = output.argmax(dim=1, keepdim=True)

    for key, val in classes.items():
        if val == pred:
            return key
    return 'None'



if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--image', type=str, default='YL (18).jpg', metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--use-cuda', action='store_true', default= False)

    # parse argument
    args = parser.parse_args()
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    model = Net().to(device)
    model.load_state_dict(torch.load(model_name))

    image = cv2.imread(args.image)

    pred_type = evaluate(image, model, device)
    print(pred_type)
    cv2.putText(image, pred_type,
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.namedWindow('Predict', cv2.WINDOW_NORMAL)
    cv2.imshow('Predict', image)
    cv2.waitKey(0)

