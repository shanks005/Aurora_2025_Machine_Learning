import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import io
import base64
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)


# Load the model (same as before)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1=nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1) #(10,28,28)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.fc1=nn.Linear(64*3*3,128) # the 64*3*3 layer is mapped to 128 nodes in the full connected layer
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,27)
    def forward(self,x):
        x=F.relu(F.max_pool2d(self.conv1(x),2))
        x=F.relu(F.max_pool2d(self.conv2(x),2))
        x=F.relu(F.max_pool2d(self.conv3(x),2))
        #flatten
        x=x.view(x.size(0),-1)
        
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model = SimpleCNN()
model.load_state_dict(torch.load('pytorch_ocr.pth', map_location=torch.device('cpu')))
model.eval()

# Mapping of class indices to letters
idx_to_letter = {i: chr(i + 65-1) for i in range(26)}
idx_to_letter[26] = ' '  # Add space as an option if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get base64 encoded image from request
    image_data = request.json['image']
    
    # Decode base64 image
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    
    # Save original image for debugging
    image.save('debug_original.png')
    
    # Invert the image (optional - test both with and without)
    # Uncomment/comment to test different preprocessing
    image = ImageOps.invert(image)
    image.save('debug_inverted.png')
    
    # Resize and preprocess
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: transforms.functional.hflip(img)),
        transforms.Lambda(lambda img: transforms.functional.rotate(img, 90)), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Debug: visualize tensor
    debug_img = input_tensor.squeeze().numpy()
    plt.imshow(debug_img, cmap='gray')
    plt.title('Preprocessed Image')
    plt.savefig('debug_tensor.png')
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        # Get top 3 predictions
        top_predictions = [
            {
                'letter': idx_to_letter[idx.item()], 
                'probability': prob.item()
            } 
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
    
    return jsonify({
        'predictions': top_predictions
    })

if __name__ == '__main__':
    app.run(debug=True)