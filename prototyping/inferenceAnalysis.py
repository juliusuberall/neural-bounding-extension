import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Loop through all saved models in folder
directory = 'prototyping/models'
files = os.listdir(directory)
files = [file for file in files if os.path.isfile(os.path.join(directory, file))]

for file in files:

    # Load model
    model = torch.load(directory + "/" + file)
    model.eval() # Set the model to evaluation mode (important for inference)

    # Create sampling tensor for model
    m, n = 32, 32
    i_values, j_values = torch.meshgrid(torch.arange(m), torch.arange(n))
    samples = torch.stack([i_values, j_values], dim=-1).reshape(-1,2)/32

    # Predict samples with model
    output = model(samples).reshape(-1).reshape(m, -1)
    output_np = output.detach().numpy()

    # Create a heatmap plot
    plt.imshow(output_np, cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)
    plt.colorbar()
    it_number = str(file.split(".")[0].split("_")[1])
    plt.title(f"Iteration {it_number}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("prototyping/certantiyMaps/" + it_number.zfill(6) + ".png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved map for: {file}")

# Create gif of inference
directory = 'prototyping/certantiyMaps'
files = os.listdir(directory)
frames = [file for file in files if os.path.isfile(os.path.join(directory, file))]
frames.sort()
frames = [Image.open('prototyping/certantiyMaps/' + frame) for frame in frames]
frames[0].save('prototyping/inferenceAnalysisResults/output.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)

# Ground truth plot
image_array = np.array(Image.open('data/2D/bunny.png'))
plt.imshow(image_array, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Ground Truth")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("prototyping/inferenceAnalysisResults/GroundTruth.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved ground truth")