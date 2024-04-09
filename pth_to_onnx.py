import torch
import torch.nn as nn
from torchvision import models


class ConvLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.vgg_backbone = models.vgg16(weights=True).features
        for param in self.vgg_backbone.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=(512*7*7) + 4, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)  # output 4 for bounding box

    def forward(self, images, bboxes):
        batch_size, sequence_length, C, H, W = images.size()
        images = images.reshape(batch_size * sequence_length, C, H, W)
        features = self.vgg_backbone(images)
        features = features.view(batch_size, sequence_length, -1)

        placeholder_bbox = torch.zeros(batch_size, 1, 4, device=images.device)
        bboxes_with_placeholder = torch.cat((bboxes, placeholder_bbox), dim=1)

        combined_input = torch.cat((features[:, :-1, :], bboxes_with_placeholder[:, :-1, :]), dim=2)
        lstm_out, _ = self.lstm(combined_input)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


# Initialize your ConvLSTM model
model_path = 'weights/model.pth'
model = ConvLSTM(hidden_size=256, num_layers=4)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to inference mode

dummy_images = torch.randn(1, 5, 3, 224, 224)
dummy_bboxes = torch.randn(1, 4, 4)


# export the model
onnx_model_path = 'weights/model.onnx'
torch.onnx.export(model,               
                  (dummy_images, dummy_bboxes),  
                  onnx_model_path,     
                  opset_version=11,    
                  do_constant_folding=True,  
                  input_names=['images', 'bboxes'],
                  output_names=['output'], 
                  dynamic_axes={'images': {0: 'batch_size'},
                                'bboxes': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})




