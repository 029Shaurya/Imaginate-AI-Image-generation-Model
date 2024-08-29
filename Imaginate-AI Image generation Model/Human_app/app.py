from flask import Flask, render_template, send_file
import torch
import random
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import os
import webbrowser

app = Flask(__name__)

latent_size = 128

generator = nn.Sequential(
    nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
)

try:
    generator.load_state_dict(torch.load('../Human_Faces/generator.pth', map_location='cpu'))
    print("Generator model loaded successfully.")
except Exception as e:
    print("Error loading generator model:", e)

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def generate_image():
    try:
        temp_latent = random.uniform(0.7, 1.3) * torch.randn(64, latent_size, 1, 1, device='cpu')
        fake_images = generator(temp_latent)
        fake_images = denorm(fake_images)
        image_grid = make_grid(fake_images.cpu().detach(), nrow=8)
        save_path = os.path.join('static', 'fake_images_grid.png')
        save_image(image_grid, save_path)
        print("Image generated and saved to", save_path)
    except Exception as e:
        print("Error generating image:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    generate_image()
    return send_file('static/fake_images_grid.png')

if __name__ == '__main__':
    app.run(debug=True)
    # Automatically open the browser
    webbrowser.open('http://127.0.0.1:5000/')
