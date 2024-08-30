import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

def save_and_open_plot(image_path = "plot.png"):
    plt.savefig(image_path)
    plt.close()

    # Open the image using the default image viewer
    if os.name == 'nt':  # For Windows
        os.startfile(image_path)
    elif os.name == 'posix':  # For macOS and Linux
        # subprocess.call(['open', image_path])  # For macOS
        subprocess.call(['xdg-open', image_path])  # For Linux

