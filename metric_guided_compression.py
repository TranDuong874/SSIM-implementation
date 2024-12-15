import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from skimage.metrics import structural_similarity as ssim, mean_squared_error, peak_signal_noise_ratio as psnr
from skimage import img_as_float
from PIL import Image
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import time

# Define compression techniques
def compress_jpeg(image, quality):
    buffer = io.BytesIO()
    pil_image = Image.fromarray((image * 255).astype("uint8"))
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return img_as_float(np.array(compressed_image.convert("L"))), buffer.getbuffer().nbytes, buffer

def compress_png(image):
    buffer = io.BytesIO()
    pil_image = Image.fromarray((image * 255).astype("uint8"))
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return img_as_float(np.array(compressed_image.convert("L"))), buffer.getbuffer().nbytes, buffer

def compress_webp(image, quality):
    buffer = io.BytesIO()
    pil_image = Image.fromarray((image * 255).astype("uint8"))
    pil_image.save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return img_as_float(np.array(compressed_image.convert("L"))), buffer.getbuffer().nbytes, buffer

# Compute SSIM
def compute_ssim(original, compressed):
    return ssim(original, compressed, data_range=1.0, win_size=11, K1=1, K2=1)

def compute_mse(original, compressed):
    return mean_squared_error(original, compressed)

def compute_psnr(original, compressed):
    return psnr(original, compressed)

quality_levels = [70]

# Main function to apply multiple compression techniques and select the best
def heuristic_compression(original_image, threshold=0.95, max_iterations=10, ssim_convergence_threshold=0.001, constraint='SSIM', priority='quality'):
    methods = [
        ('JPEG', compress_jpeg, quality_levels), 
        ('PNG', compress_png, [None]),  
        ('WEBP', compress_webp, quality_levels)
    ]

    if constraint == 'SSIM':
        compute_metric = compute_ssim
        target_is_lower_better = False
    elif constraint == 'MSE':
        compute_metric = compute_mse
        target_is_lower_better = True
    elif constraint == 'PSNR':
        compute_metric = compute_psnr
        target_is_lower_better = False
    else:
        raise ValueError("Invalid constraint type. Choose 'SSIM', 'MSE', or 'PSNR'.")

    best_compressed_image = original_image
    best_size = float('inf')
    best_metric = 0 if not target_is_lower_better else float('inf')
    best_method = None
    best_quality = None
    best_buffer = None
    compression_history = []
    current_image = original_image
    continue_compression = True
    iteration_count = 0
    last_metric = 0

    while continue_compression and iteration_count < max_iterations:
        best_step_image = None
        best_step_size = float('inf')
        best_step_metric = 0 if not target_is_lower_better else float('inf')
        best_step_method = None
        best_step_quality = None
        best_step_buffer = None
        continue_compression = False

        # Try each method with its corresponding parameters
        for method_name, method_func, qualities in methods:
            for quality in qualities:
                # Compress the image
                if quality is not None:
                    compressed_image, file_size, buffer = method_func(current_image, quality)
                else:
                    compressed_image, file_size, buffer = method_func(current_image)

                # Compute the metric for the compressed image
                current_metric = compute_metric(original_image, compressed_image)

                # Log the history of compression steps
                compression_history.append({
                    "method": method_name,
                    "quality": quality if quality is not None else "N/A",
                    "size": file_size,
                    "metric": current_metric
                })

                # Update the history text widget
                history_text.insert(tk.END, f"Iteration {iteration_count + 1}: Method: {method_name}, Quality: {quality}, Size: {file_size} bytes, Metric: {current_metric:.4f}\n")
                root.update_idletasks()

                # Check if the metric is above/below threshold and if the size is smaller or the quality is higher
                if ((not target_is_lower_better and current_metric >= threshold) or 
                    (target_is_lower_better and current_metric <= threshold)) and \
                    ((priority == 'quality' and current_metric > best_step_metric and file_size <= best_step_size) or \
                    (priority == 'size' and file_size < best_step_size)):
                    best_step_image = compressed_image
                    best_step_size = file_size
                    best_step_metric = current_metric
                    best_step_method = method_name
                    best_step_quality = quality
                    best_step_buffer = buffer
                    continue_compression = True

        # Update the best compression result if we found a better one in this step
        if best_step_image is not None and best_step_size < best_size:
            best_compressed_image = best_step_image
            best_size = best_step_size
            best_metric = best_step_metric
            best_method = best_step_method
            best_quality = best_step_quality
            best_buffer = best_step_buffer

            # Update the current image for the next iteration
            current_image = best_step_image

        # Check for metric convergence
        if abs(best_metric - last_metric) < ssim_convergence_threshold:
            break

        last_metric = best_metric
        iteration_count += 1

    return best_compressed_image, best_method, best_quality, best_size, best_metric, compression_history, best_buffer

# Load image function
def load_image(file_path):
    img = Image.open(file_path).convert("L")  # Convert to grayscale
    return img_as_float(np.array(img))

# GUI Setup
root = tk.Tk()
root.title("Metric-Guided Image Compression - Mgic")
root.geometry("800x600")

header_label = tk.Label(root, text="Mgic", font=("Arial", 18))
header_label.pack(pady=20)

instructions = tk.Label(root, text="Select an image")
instructions.pack()

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not file_path:
        return None
    img = load_image(file_path)
    return img, file_path

def open_folder():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return None
    return folder_path

def save_compressed_image(buffer, original_path):
    save_path = filedialog.asksaveasfilename(
        defaultextension="." + original_path.split('.')[-1],
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp")]
    )
    if save_path:
        with open(save_path, "wb") as f:
            f.write(buffer.getbuffer())
        messagebox.showinfo("Image Saved", f"Compressed image saved to {save_path}")

def start_single_image_demo():
    img, file_path = open_file()
    if img is None:
        messagebox.showwarning("No File Selected", "Please select a valid image file.")
        return

    threshold = float(threshold_entry.get())
    max_iterations = int(iteration_entry.get())
    constraint_type = constraint_var.get()
    priority_type = priority_var.get()
    
    # Clear the history text widget before starting the compression
    history_text.delete(1.0, tk.END)

    start = time.time()
    compressed_image, method, quality, size, metric_value, compression_history, best_buffer = heuristic_compression(
        img, threshold, max_iterations, constraint=constraint_type, priority=priority_type)
    end = time.time()
    
    original_size = os.path.getsize(file_path)
    reduction_percentage = (1 - size / original_size) * 100
    messagebox.showinfo("Compression Complete",f"Operation took \n{(end - start)} seconds \nBest Method: {method}\nQuality: {quality}\nSize: {size} bytes\n{constraint_type}: {metric_value}\nReduction: {reduction_percentage:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Original\nSize: {original_size // 1024} KB")
    axes[0].axis("off")

    axes[1].imshow(compressed_image, cmap="gray")
    axes[1].set_title(f"Compressed (Q={quality})\nSize: {size // 1024} KB\n{constraint_type}: {metric_value:.4f}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Reset the save button for the new image
    save_button.configure(command=lambda: save_compressed_image(best_buffer, file_path))

threshold_label = tk.Label(root, text="Threshold")
threshold_label.pack()
threshold_entry = tk.Entry(root)
threshold_entry.insert(0, "0.99")
threshold_entry.pack()

iteration_label = tk.Label(root, text="Max Iterations")
iteration_label.pack()
iteration_entry = tk.Entry(root)
iteration_entry.insert(0, "10")
iteration_entry.pack()

constraint_label = tk.Label(root, text="Constraint Type")
constraint_label.pack()
constraint_var = tk.StringVar(value="SSIM")
constraint_menu = ttk.Combobox(root, textvariable=constraint_var, values=["SSIM", "MSE", "PSNR"])
constraint_menu.pack()

priority_label = tk.Label(root, text="Priority")
priority_label.pack()
priority_var = tk.StringVar(value="quality")
priority_menu = ttk.Combobox(root, textvariable=priority_var, values=["quality", "size"])
priority_menu.pack()

single_image_button = tk.Button(root, text="Choose Single Image", font=("Arial", 14), command=start_single_image_demo)
single_image_button.pack(pady=20)

history_label = tk.Label(root, text="Compression History:", font=("Arial", 12))
history_label.pack(pady=10)
history_text = tk.Text(root, height=10, width=70)
history_text.pack(pady=10)

save_button = tk.Button(root, text="Save Compressed Image")
save_button.pack(pady=10)

root.mainloop()
