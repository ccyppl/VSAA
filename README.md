# VSAA
variable-scale alignment algorithm (VSAA)
This is the repository of variable-scale alignment algorithm (VSAA). To run the model, you should first install the required packages through pip install -r requirments.txt.
# Environment
Python 3.10 + Torch 2.4.1 + CUDA 11.8 conda create --name VSAA python=3.10 conda activate VSAA
# Hardware:
single RTX 4090 GPU
# Run the model
The axle box acceleration is used to calculate the Track Impact Index(TII) and find feature points from the axle box acceleration data.

VSAA/Acceleration detection/TI_track.py
VSAA/Acceleration detection/find_peak.py

The Image detection is used to detect rail joints in images.

Install the ultralytics package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8.

pip install ultralytics

The Alignment Algorithm contains the alignment algorithm code. 

Run the VSAA/Alignment Algorithm/Changescale-RC.py code.

