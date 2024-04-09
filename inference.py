import os
import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from network import ConvLSTM

root_dir = 'data'

# load images here
all_files = os.listdir(root_dir)
image_files_count = len([file for file in all_files if file.endswith('.jpg')])

image_paths= [os.path.join(root_dir, f'{index + 1:08}.jpg') for index in range(image_files_count)]
frames = [cv2.imread(path) for path in image_paths]



#please provide four bboxes for -> first four frames, then starts online
annotations  = np.array([[6.000000,166.000000,49.000000,193.000000],
                        [7.000000,168.000000,50.000000,192.000000],
                        [10.000000,168.000000,54.000000,192.000000],
                        [12.000000,166.000000,54.000000,192.000000]])

# load the model
model = ConvLSTM(hidden_size=256, num_layers=4)
model.load_state_dict(torch.load('weights/model.pth'))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transformation pipeline for the images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prepare_initial_bboxes(annotations, original_size=(272, 640), new_size=(224, 224)):
    resized_bboxes = []
    for bbox in annotations[:4]:  # want the first 4 bounding boxes
        # resize bbox to img dimensions
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        resized_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
        
        # normalize resized bbox
        normalized_bbox = [coord / new_size[0] for coord in resized_bbox]
        resized_bboxes.append(normalized_bbox)
    
    return resized_bboxes

initial_bboxes = prepare_initial_bboxes(annotations)


# function to predict the bounding box for the next frame
def predict_next_bbox(model, frames, prev_bboxes):
    frames_tensor = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(device)
    bboxes_tensor = torch.tensor(prev_bboxes, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_bbox = model(frames_tensor, bboxes_tensor)
    
    return pred_bbox.squeeze().cpu().numpy()



prev_time = time.time()  #for fps calc

for i in range(4, len(frames)):
    # get the next sequence of 5 frames
    sequence = frames[i-4:i+1]  # Current sequence of 5 frames
    
    # predict bounding box for the fifth frame
    pred_bbox = predict_next_bbox(model, sequence, initial_bboxes)
    
    # update initial_bboxes for the next iteration
    initial_bboxes.append(pred_bbox.tolist())
    initial_bboxes.pop(0)  # remove the oldest bounding box
    
    # fps calc
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # display the last frame with the predicted bounding box and FPS
    last_frame = sequence[-1].copy()
    h, w, _ = last_frame.shape
    pt1 = (int(pred_bbox[0] * w), int(pred_bbox[1] * h))
    pt2 = (int(pred_bbox[2] * w), int(pred_bbox[3] * h))
    cv2.rectangle(last_frame, pt1, pt2, (0, 255, 0), 2)
    
    # display fpss
    cv2.putText(last_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Prediction", last_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
