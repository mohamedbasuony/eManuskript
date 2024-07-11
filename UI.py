import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, Toplevel, messagebox
import cv2
import numpy as np
from sklearn.decomposition import PCA

# Initialize the main window
root = ctk.CTk()
root.state('zoomed')  # Set the window to full screen
root.title("AppName")

# Set the appearance mode of the app to 'dark'
ctk.set_appearance_mode("dark")

# Load the logo image
logo_path = "/Users/mohamedbasuony/Downloads/HokuLike Logo.png"  # Replace with the path to your logo image
logo_image = Image.open(logo_path)
logo_image = logo_image.resize((100, 100), Image.LANCZOS)  # Resize the image using LANCZOS filter
logo_photo = ImageTk.PhotoImage(logo_image)


def generate_pseudocolor_effect(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        raise ValueError("One or both images could not be loaded.")

    # Ensure the images have the same size
    if image1.shape != image2.shape:
        raise ValueError("The images must have the same size.")

    # Combine the images
    combined_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

    # Apply the colormap to create the pseudocolor effect
    pseudocolor_image = cv2.applyColorMap(combined_image, cv2.COLORMAP_JET)

    return pseudocolor_image


def pseudocolor():
    def open_file(entry):
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.configure(state='normal')
            entry.delete(0, ctk.END)
            entry.insert(0, file_path)
            entry.configure(state='readonly')

    def cancel():
        popup.destroy()

    def generate_effect():
        image1_path = entry_image1.get()
        image2_path = entry_image2.get()
        if image1_path and image2_path:
            try:
                # Generate the pseudocolor effect here
                pseudocolor_image = generate_pseudocolor_effect(image1_path, image2_path)

                # Convert the processed image to PIL format for displaying in Tkinter
                pseudocolor_image_rgb = cv2.cvtColor(pseudocolor_image, cv2.COLOR_BGR2RGB)
                pseudocolor_image_pil = Image.fromarray(pseudocolor_image_rgb)

                # Resize the image to fit within the main window
                max_width = root.winfo_width() - 40  # Leave some padding
                max_height = root.winfo_height() - 200  # Leave some padding for the buttons
                pseudocolor_image_pil.thumbnail((max_width, max_height), Image.LANCZOS)

                pseudocolor_image_tk = ImageTk.PhotoImage(pseudocolor_image_pil)

                # Close the popup
                popup.destroy()

                # Remove the welcome label if it exists
                if label.winfo_exists():
                    label.pack_forget()

                # Display the pseudocolor image in the main window
                result_label.configure(image=pseudocolor_image_tk, text="")
                result_label.image = pseudocolor_image_tk
                result_label.current_image_pil = pseudocolor_image_pil

            except ValueError as e:
                messagebox.showerror("Error", str(e))

    # Calculate the position to center the popup window
    popup_width = 500
    popup_height = 350
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    popup_x = int((screen_width - popup_width) / 2)
    popup_y = int((screen_height - popup_height) / 2)

    popup = Toplevel(root)
    popup.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
    popup.title("Upload Images for Pseudocolor Effect")

    frame = ctk.CTkFrame(popup)
    frame.pack(fill="both", expand=True)

    label_instruction = ctk.CTkLabel(frame, text="Please upload UV and IR images:", font=("Helvetica", 14))
    label_instruction.pack(pady=10)

    frame_inputs = ctk.CTkFrame(frame)
    frame_inputs.pack(pady=10, padx=10, fill="x")

    label_image1 = ctk.CTkLabel(frame_inputs, text="UV Image:")
    label_image1.grid(row=0, column=0, pady=5, padx=5, sticky="w")
    entry_image1 = ctk.CTkEntry(frame_inputs, width=250)
    entry_image1.grid(row=0, column=1, pady=5, padx=5)
    button_browse1 = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image1))
    button_browse1.grid(row=0, column=2, pady=5, padx=5)

    label_image2 = ctk.CTkLabel(frame_inputs, text="IR Image:")
    label_image2.grid(row=1, column=0, pady=5, padx=5, sticky="w")
    entry_image2 = ctk.CTkEntry(frame_inputs, width=250)
    entry_image2.grid(row=1, column=1, pady=5, padx=5)
    button_browse2 = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image2))
    button_browse2.grid(row=1, column=2, pady=5, padx=5)

    frame_actions = ctk.CTkFrame(frame)
    frame_actions.pack(pady=20)

    button_cancel = ctk.CTkButton(frame_actions, text="Cancel", command=cancel, width=100)
    button_cancel.grid(row=0, column=0, padx=10)

    button_generate = ctk.CTkButton(frame_actions, text="Generate", command=generate_effect, width=100)
    button_generate.grid(row=0, column=1, padx=10)


# Function to create and display the popup window for Sharpie effect
def sharpie_effect(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    dilated_edges = cv2.dilate(edges, kernel=np.ones((3, 3), np.uint8), iterations=1)

    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    color_edges = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

    sharpie_image = cv2.addWeighted(image, 0.7, color_edges, 0.3, 0)

    return sharpie_image


# Function to create and display the popup window for the Sharpie effect
def sharpie():
    def open_file(entry):
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.configure(state='normal')
            entry.delete(0, ctk.END)
            entry.insert(0, file_path)
            entry.configure(state='readonly')

    def cancel():
        popup.destroy()

    def generate_effect():
        image_path = entry_image.get()
        if image_path:
            try:
                sharpie_image = sharpie_effect(image_path)

                # Convert the processed image to PIL format for displaying in Tkinter
                sharpie_image_rgb = cv2.cvtColor(sharpie_image, cv2.COLOR_BGR2RGB)
                sharpie_image_pil = Image.fromarray(sharpie_image_rgb)

                # Resize the image to fit within the main window
                max_width = root.winfo_width() - 40  # Leave some padding
                max_height = root.winfo_height() - 200  # Leave some padding for the buttons
                sharpie_image_pil.thumbnail((max_width, max_height), Image.LANCZOS)

                sharpie_image_tk = ImageTk.PhotoImage(sharpie_image_pil)

                # Close the popup
                popup.destroy()

                # Remove the welcome label if it exists
                if label.winfo_exists():
                    label.pack_forget()

                # Display the Sharpie effect image in the main window
                result_label.configure(image=sharpie_image_tk, text="")
                result_label.image = sharpie_image_tk
                result_label.current_image_pil = sharpie_image_pil

            except ValueError as e:
                print(e)

    # Calculate the position to center the popup window
    popup_width = 500
    popup_height = 250
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    popup_x = int((screen_width - popup_width) / 2)
    popup_y = int((screen_height - popup_height) / 2)

    popup = Toplevel(root)
    popup.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
    popup.title("Upload Image for Sharpie Effect")

    frame = ctk.CTkFrame(popup)
    frame.pack(fill="both", expand=True)

    label_instruction = ctk.CTkLabel(frame, text="Please upload an image:", font=("Helvetica", 14))
    label_instruction.pack(pady=10)

    frame_inputs = ctk.CTkFrame(frame)
    frame_inputs.pack(pady=10, padx=10, fill="x")

    label_image = ctk.CTkLabel(frame_inputs, text="Image:")
    label_image.grid(row=0, column=0, pady=5, padx=5, sticky="w")
    entry_image = ctk.CTkEntry(frame_inputs, width=250)
    entry_image.grid(row=0, column=1, pady=5, padx=5)
    button_browse = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image))
    button_browse.grid(row=0, column=2, pady=5, padx=5)

    frame_actions = ctk.CTkFrame(frame)
    frame_actions.pack(pady=20)

    button_cancel = ctk.CTkButton(frame_actions, text="Cancel", command=cancel, width=100)
    button_cancel.grid(row=0, column=0, padx=10)

    button_generate = ctk.CTkButton(frame_actions, text="Generate", command=generate_effect, width=100)
    button_generate.grid(row=0, column=1, padx=10)


# Function to handle the quit event
def on_quit():
    root.quit()


# Function to raise image to a given power
def raise_image_to_power(image_path, power):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    image_float = image.astype(np.float32) / 255.0
    image_power = np.power(image_float, power)
    image_power = np.clip(image_power * 255, 0, 255).astype(np.uint8)

    return image_power


# Function for partial inversion
def partial_inversion(image_path, alpha):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Invert the image using bitwise NOT
    inverted_image = cv2.bitwise_not(image)

    # Blend the original and inverted images
    output_image = cv2.addWeighted(image, 1 - alpha, inverted_image, alpha, 0)

    return output_image


# Function to create and display the popup window for the power effect
def power():
    def open_file(entry):
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.configure(state='normal')
            entry.delete(0, ctk.END)
            entry.insert(0, file_path)
            entry.configure(state='readonly')

    def cancel():
        popup.destroy()

    def generate_effect():
        image_path = entry_image.get()
        power_value = entry_power.get()
        if image_path and power_value:
            try:
                power_value = float(power_value)
                if power_value < 0:
                    raise ValueError("Power must be a positive number.")

                power_image = raise_image_to_power(image_path, power_value)

                # Convert the processed image to PIL format for displaying in Tkinter
                power_image_rgb = cv2.cvtColor(power_image, cv2.COLOR_BGR2RGB)
                power_image_pil = Image.fromarray(power_image_rgb)

                # Resize the image to fit within the main window
                max_width = root.winfo_width() - 40  # Leave some padding
                max_height = root.winfo_height() - 300  # Leave some padding for the buttons
                power_image_pil.thumbnail((max_width, max_height), Image.LANCZOS)

                power_image_tk = ImageTk.PhotoImage(power_image_pil)

                # Close the popup
                popup.destroy()

                # Remove the welcome label if it exists
                if label.winfo_exists():
                    label.pack_forget()

                # Display the power effect image in the main window
                result_label.configure(image=power_image_tk, text="")
                result_label.image = power_image_tk
                result_label.current_image_pil = power_image_pil  # Save the PIL image for rotation

                # Center the image below the top frame
                result_label.pack(pady=20)

            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))

    # Calculate the position to center the popup window
    popup_width = 500
    popup_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    popup_x = int((screen_width - popup_width) / 2)
    popup_y = int((screen_height - popup_height) / 2)

    popup = Toplevel(root)
    popup.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
    popup.title("Upload Image for Power Effect")

    frame = ctk.CTkFrame(popup)
    frame.pack(fill="both", expand=True)

    label_instruction = ctk.CTkLabel(frame, text="Please upload an image and enter a power value:",
                                     font=("Helvetica", 14))
    label_instruction.pack(pady=10)

    frame_inputs = ctk.CTkFrame(frame)
    frame_inputs.pack(pady=10, padx=10, fill="x")

    label_image = ctk.CTkLabel(frame_inputs, text="Image:")
    label_image.grid(row=0, column=0, pady=5, padx=5, sticky="w")
    entry_image = ctk.CTkEntry(frame_inputs, width=250)
    entry_image.grid(row=0, column=1, pady=5, padx=5)
    button_browse = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image))
    button_browse.grid(row=0, column=2, pady=5, padx=5)

    label_power = ctk.CTkLabel(frame_inputs, text="Power:")
    label_power.grid(row=1, column=0, pady=5, padx=5, sticky="w")
    entry_power = ctk.CTkEntry(frame_inputs, width=250)
    entry_power.grid(row=1, column=1, pady=5, padx=5)

    frame_actions = ctk.CTkFrame(frame)
    frame_actions.pack(pady=20)

    button_cancel = ctk.CTkButton(frame_actions, text="Cancel", command=cancel, width=100)
    button_cancel.grid(row=0, column=0, padx=10)

    button_generate = ctk.CTkButton(frame_actions, text="Generate", command=generate_effect, width=100)
    button_generate.grid(row=0, column=1, padx=10)


# Function to create and display the popup window for the invert effect
def invert():
    def open_file(entry):
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.configure(state='normal')
            entry.delete(0, ctk.END)
            entry.insert(0, file_path)
            entry.configure(state='readonly')

    def cancel():
        popup.destroy()

    def generate_effect():
        image_path = entry_image.get()
        alpha_value = entry_alpha.get()
        if image_path and alpha_value:
            try:
                alpha_value = float(alpha_value)
                if not (0 <= alpha_value <= 1):
                    raise ValueError("Alpha must be between 0 and 1.")

                inverted_image = partial_inversion(image_path, alpha_value)

                # Convert the processed image to PIL format for displaying in Tkinter
                inverted_image_pil = Image.fromarray(inverted_image)

                # Resize the image to fit within the main window
                max_width = root.winfo_width() - 40  # Leave some padding
                max_height = root.winfo_height() - 300  # Leave some padding for the buttons
                inverted_image_pil.thumbnail((max_width, max_height), Image.LANCZOS)

                inverted_image_tk = ImageTk.PhotoImage(inverted_image_pil)

                # Close the popup
                popup.destroy()

                # Remove the welcome label if it exists
                if label.winfo_exists():
                    label.pack_forget()

                # Display the inverted effect image in the main window
                result_label.configure(image=inverted_image_tk, text="")
                result_label.image = inverted_image_tk
                result_label.current_image_pil = inverted_image_pil  # Save the PIL image for rotation

                # Center the image below the top frame
                result_label.pack(pady=20)

            except ValueError as e:
                messagebox.showerror("Invalid Input", str(e))

    # Calculate the position to center the popup window
    popup_width = 500
    popup_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    popup_x = int((screen_width - popup_width) / 2)
    popup_y = int((screen_height - popup_height) / 2)

    popup = Toplevel(root)
    popup.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
    popup.title("Upload Image for Partial Inversion")

    frame = ctk.CTkFrame(popup)
    frame.pack(fill="both", expand=True)

    label_instruction = ctk.CTkLabel(frame, text="Please upload an image and enter an alpha value:",
                                     font=("Helvetica", 14))
    label_instruction.pack(pady=10)

    frame_inputs = ctk.CTkFrame(frame)
    frame_inputs.pack(pady=10, padx=10, fill="x")

    label_image = ctk.CTkLabel(frame_inputs, text="Image:")
    label_image.grid(row=0, column=0, pady=5, padx=5, sticky="w")
    entry_image = ctk.CTkEntry(frame_inputs, width=250)
    entry_image.grid(row=0, column=1, pady=5, padx=5)
    button_browse = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image))
    button_browse.grid(row=0, column=2, pady=5, padx=5)

    label_alpha = ctk.CTkLabel(frame_inputs, text="Alpha:")
    label_alpha.grid(row=1, column=0, pady=5, padx=5, sticky="w")
    entry_alpha = ctk.CTkEntry(frame_inputs, width=250)
    entry_alpha.grid(row=1, column=1, pady=5, padx=5)

    frame_actions = ctk.CTkFrame(frame)
    frame_actions.pack(pady=20)

    button_cancel = ctk.CTkButton(frame_actions, text="Cancel", command=cancel, width=100)
    button_cancel.grid(row=0, column=0, padx=10)

    button_generate = ctk.CTkButton(frame_actions, text="Generate", command=generate_effect, width=100)
    button_generate.grid(row=0, column=1, padx=10)


def rotateRight():
    if result_label.image:
        # Rotate the image to the right by 90 degrees
        rotated_image = result_label.current_image_pil.rotate(-90, expand=True)
        rotated_image_tk = ImageTk.PhotoImage(rotated_image)

        # Update the displayed image
        result_label.configure(image=rotated_image_tk)
        result_label.image = rotated_image_tk
        result_label.current_image_pil = rotated_image  # Save the rotated image for further rotations


def rotateLeft():
    if result_label.image:
        # Rotate the image to the left by 90 degrees
        rotated_image = result_label.current_image_pil.rotate(90, expand=True)
        rotated_image_tk = ImageTk.PhotoImage(rotated_image)

        # Update the displayed image
        result_label.configure(image=rotated_image_tk)
        result_label.image = rotated_image_tk
        result_label.current_image_pil = rotated_image  # Save the rotated image for further rotations


def save():
    print("Save")


def import_file():
    print("Import")


def blur_and_divide(image, blur_value):
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (blur_value, blur_value), 0)

    # Convert the images to float32 for precision during division
    original_float = image.astype(np.float32)
    blurred_float = blurred_image.astype(np.float32)

    # Avoid division by zero by adding a small epsilon to the blurred image
    epsilon = 1e-10
    blurred_float = np.clip(blurred_float, epsilon, None)

    # Divide the original image by the blurred image
    divided_image = original_float / blurred_float

    # Normalize the resulting image to the range [0, 255]
    divided_image_normalized = cv2.normalize(divided_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert back to uint8
    divided_image_uint8 = divided_image_normalized.astype(np.uint8)

    # Apply histogram equalization to enhance contrast
    if len(divided_image_uint8.shape) == 2:  # Grayscale image
        equalized_image = cv2.equalizeHist(divided_image_uint8)
    else:  # Color image
        equalized_image = cv2.cvtColor(divided_image_uint8, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(equalized_image)
        cv2.equalizeHist(channels[0], channels[0])
        equalized_image = cv2.merge(channels)
        equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCrCb2BGR)

    return blurred_image, divided_image_uint8, equalized_image


def blur_and_divide_effect():
    def open_file(entry):
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.configure(state='normal')
            entry.delete(0, ctk.END)
            entry.insert(0, file_path)
            entry.configure(state='readonly')

    def cancel():
        popup.destroy()

    def generate_effect():
        image_path = entry_image.get()
        if image_path:
            try:
                original_image = cv2.imread(image_path)
                if original_image is None:
                    raise ValueError(f"Image at {image_path} could not be loaded.")

                blur_value = 15  # You can adjust the blur value as needed
                blurred_image, divided_image, equalized_image = blur_and_divide(original_image, blur_value)

                # Convert the processed image to PIL format for displaying in Tkinter
                equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
                equalized_image_pil = Image.fromarray(equalized_image_rgb)

                # Resize the image to fit within the main window
                max_width = root.winfo_width() - 40  # Leave some padding
                max_height = root.winfo_height() - 200  # Leave some padding for the buttons
                equalized_image_pil.thumbnail((max_width, max_height), Image.LANCZOS)

                equalized_image_tk = ImageTk.PhotoImage(equalized_image_pil)

                # Close the popup
                popup.destroy()

                # Remove the welcome label if it exists
                if label.winfo_exists():
                    label.pack_forget()

                # Display the equalized image in the main window
                result_label.configure(image=equalized_image_tk, text="")
                result_label.image = equalized_image_tk
                result_label.current_image_pil = equalized_image_pil  # Save the PIL image for rotation

            except ValueError as e:
                print(e)

    # Calculate the position to center the popup window
    popup_width = 500
    popup_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    popup_x = int((screen_width - popup_width) / 2)
    popup_y = int((screen_height - popup_height) / 2)

    popup = Toplevel(root)
    popup.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
    popup.title("Upload Image for Blur and Divide Effect")

    frame = ctk.CTkFrame(popup)
    frame.pack(fill="both", expand=True)

    label_instruction = ctk.CTkLabel(frame, text="Please upload an image:",
                                     font=("Helvetica", 14))
    label_instruction.pack(pady=10)

    frame_inputs = ctk.CTkFrame(frame)
    frame_inputs.pack(pady=10, padx=10, fill="x")

    label_image = ctk.CTkLabel(frame_inputs, text="Image:")
    label_image.grid(row=0, column=0, pady=5, padx=5, sticky="w")
    entry_image = ctk.CTkEntry(frame_inputs, width=250)
    entry_image.grid(row=0, column=1, pady=5, padx=5)
    button_browse = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image))
    button_browse.grid(row=0, column=2, pady=5, padx=5)

    frame_actions = ctk.CTkFrame(frame)
    frame_actions.pack(pady=20)

    button_cancel = ctk.CTkButton(frame_actions, text="Cancel", command=cancel, width=100)
    button_cancel.grid(row=0, column=0, padx=10)

    button_generate = ctk.CTkButton(frame_actions, text="Generate", command=generate_effect, width=100)
    button_generate.grid(row=0, column=1, padx=10)


def pca_transform(image_uv_path, image_lime_path, image_ir_path):
    # Load the three images with effects
    image_uv = cv2.imread(image_uv_path)
    image_lime = cv2.imread(image_lime_path)
    image_ir = cv2.imread(image_ir_path)

    # Define the effects colors
    effects = {'UV': [0, 0, 255], 'Lime': [0, 255, 0], 'IR': [255, 0, 0]}

    # Initialize an array for the combined image (assuming all images have the same size)
    combined_image = np.zeros_like(image_uv)

    # Apply effects and combine images
    for effect, color in effects.items():
        if effect == 'UV':
            image_effect = cv2.addWeighted(image_uv, 0.5, np.full(image_uv.shape, color, dtype=np.uint8), 0.5, 0)
        elif effect == 'Lime':
            image_effect = cv2.addWeighted(image_lime, 0.5, np.full(image_lime.shape, color, dtype=np.uint8), 0.5, 0)
        elif effect == 'IR':
            image_effect = cv2.addWeighted(image_ir, 0.5, np.full(image_ir.shape, color, dtype=np.uint8), 0.5, 0)

        # Add the effect to the combined image
        combined_image = cv2.add(combined_image, image_effect)

    # Perform PCA on the combined image
    pca = PCA(n_components=1)  # Keep 1 principal component
    combined_image_flat = combined_image.reshape(-1, 1)
    components = pca.fit_transform(combined_image_flat)

    # Reshape the transformed components back to the original image shape
    transformed_image = components.reshape(combined_image.shape)

    # Scale the transformed image components to the range [0, 255]
    transformed_image = cv2.normalize(transformed_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return transformed_image


def pca_effect():
    def open_file(entry):
        file_path = filedialog.askopenfilename()
        if file_path:
            entry.configure(state='normal')
            entry.delete(0, ctk.END)
            entry.insert(0, file_path)
            entry.configure(state='readonly')

    def cancel():
        popup.destroy()

    def generate_effect():
        image_uv_path = entry_image_uv.get()
        image_lime_path = entry_image_lime.get()
        image_ir_path = entry_image_ir.get()
        if image_uv_path and image_lime_path and image_ir_path:
            try:
                transformed_image = pca_transform(image_uv_path, image_lime_path, image_ir_path)

                # Convert the processed image to PIL format for displaying in Tkinter
                transformed_image_pil = Image.fromarray(transformed_image)

                # Resize the image to fit within the main window
                max_width = root.winfo_width() - 40  # Leave some padding
                max_height = root.winfo_height() - 200  # Leave some padding for the buttons
                transformed_image_pil.thumbnail((max_width, max_height), Image.LANCZOS)

                transformed_image_tk = ImageTk.PhotoImage(transformed_image_pil)

                # Close the popup
                popup.destroy()

                # Remove the welcome label if it exists
                if label.winfo_exists():
                    label.pack_forget()

                # Display the PCA transformed image in the main window
                result_label.configure(image=transformed_image_tk, text="")
                result_label.image = transformed_image_tk
                result_label.current_image_pil = transformed_image_pil  # Save the PIL image for rotation

            except ValueError as e:
                print(e)
# Calculate the position to center the popup window
    popup_width = 500
    popup_height = 350
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    popup_x = int((screen_width - popup_width) / 2)
    popup_y = int((screen_height - popup_height) / 2)

    popup = Toplevel(root)
    popup.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
    popup.title("Upload Images for PCA Effect")

    frame = ctk.CTkFrame(popup)
    frame.pack(fill="both", expand=True)

    label_instruction = ctk.CTkLabel(frame, text="Please upload your images:", font=("Helvetica", 14))
    label_instruction.pack(pady=10)

    frame_inputs = ctk.CTkFrame(frame)
    frame_inputs.pack(pady=10, padx=10, fill="x")

    label_image_uv = ctk.CTkLabel(frame_inputs, text="Image 1:")
    label_image_uv.grid(row=0, column=0, pady=5, padx=5, sticky="w")
    entry_image_uv = ctk.CTkEntry(frame_inputs, width=250)
    entry_image_uv.grid(row=0, column=1, pady=5, padx=5)
    button_browse_uv = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image_uv))
    button_browse_uv.grid(row=0, column=2, pady=5, padx=5)

    label_image_lime = ctk.CTkLabel(frame_inputs, text="Image 2:")
    label_image_lime.grid(row=1, column=0, pady=5, padx=5, sticky="w")
    entry_image_lime = ctk.CTkEntry(frame_inputs, width=250)
    entry_image_lime.grid(row=1, column=1, pady=5, padx=5)
    button_browse_lime = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image_lime))
    button_browse_lime.grid(row=1, column=2, pady=5, padx=5)

    label_image_ir = ctk.CTkLabel(frame_inputs, text="Image 3:")
    label_image_ir.grid(row=2, column=0, pady=5, padx=5, sticky="w")
    entry_image_ir = ctk.CTkEntry(frame_inputs, width=250)
    entry_image_ir.grid(row=2, column=1, pady=5, padx=5)
    button_browse_ir = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image_ir))
    button_browse_ir.grid(row=2, column=2, pady=5, padx=5)

    frame_actions = ctk.CTkFrame(frame)
    frame_actions.pack(pady=20)

    button_cancel = ctk.CTkButton(frame_actions, text="Cancel", command=cancel, width=100)
    button_cancel.grid(row=0, column=0, padx=10)

    button_generate = ctk.CTkButton(frame_actions, text="Generate", command=generate_effect, width=100)
    button_generate.grid(row=0, column=1, padx=10)


# Create a frame for the top section
top_frame = ctk.CTkFrame(root)
top_frame.pack(side="top", fill="x", pady=10)

# Add the logo to the top-left corner
logo_label = ctk.CTkLabel(top_frame, image=logo_photo, text="")
logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

# Create buttons for effects
buttons_frame = ctk.CTkFrame(top_frame)
buttons_frame.grid(row=0, column=1, padx=10)

button_effect1 = ctk.CTkButton(buttons_frame, text="Pseudocolor", command=pseudocolor, width=100, height=40)
button_effect2 = ctk.CTkButton(buttons_frame, text="Sharpie", command=sharpie, width=100, height=40)
button_effect3 = ctk.CTkButton(buttons_frame, text="Power", command=power, width=100, height=40)
button_effect4 = ctk.CTkButton(buttons_frame, text="Invert", command=invert, width=100, height=40)
button_effect5 = ctk.CTkButton(buttons_frame, text="Rotate Right", command=rotateRight, width=100, height=40)
button_effect6 = ctk.CTkButton(buttons_frame, text="Rotate Left", command=rotateLeft, width=100, height=40)
button_effect7 = ctk.CTkButton(buttons_frame, text="Blur & Divide", command=blur_and_divide_effect, width=100,
                               height=40)
button_effect8 = ctk.CTkButton(buttons_frame, text="PCA", command=pca_effect, width=100, height=40)
button_effect9 = ctk.CTkButton(buttons_frame, text="SAM", width=100, height=40)

button_effect1.grid(row=0, column=0, padx=5)
button_effect2.grid(row=0, column=1, padx=5)
button_effect3.grid(row=0, column=2, padx=5)
button_effect4.grid(row=0, column=3, padx=5)
button_effect5.grid(row=0, column=4, padx=5)
button_effect6.grid(row=0, column=5, padx=5)
button_effect7.grid(row=0, column=6, padx=5)
button_effect8.grid(row=0, column=7, padx=5)
button_effect9.grid(row=0, column=8, padx=5)

# Create buttons for save and import
button_save = ctk.CTkButton(top_frame, text="Save", command=save, width=100, height=40)
button_import = ctk.CTkButton(top_frame, text="Import", command=import_file, width=100, height=40)

button_save.grid(row=0, column=2, padx=5, sticky="e")
button_import.grid(row=0, column=3, padx=5, sticky="e")

top_frame.grid_columnconfigure(0, weight=0)  # Logo column
top_frame.grid_columnconfigure(1, weight=1)  # Buttons column
top_frame.grid_columnconfigure(2, weight=0)  # Save and Import column

# Create a welcome label in the center
label = ctk.CTkLabel(root, text="Welcome to AppName", font=("Helvetica", 24))
label.pack(pady=20)

# Create a label to display the effect image
result_label = ctk.CTkLabel(root, text="")
result_label.pack(pady=20)

# Start the main loop
root.mainloop()
