import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, Toplevel
import cv2
import numpy as np

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

# Function to create and display the popup window
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
            # Load the original UV and red/IR images in color
            uv_image_color = cv2.imread(image1_path)
            red_ir_image_color = cv2.imread(image2_path)

            if uv_image_color is None:
                print("UV image could not be loaded.")
                return
            if red_ir_image_color is None:
                print("Red/IR image could not be loaded.")
                return

            # Resize images to the same dimensions (resize to the smaller dimensions)
            height = min(uv_image_color.shape[0], red_ir_image_color.shape[0])
            width = min(uv_image_color.shape[1], red_ir_image_color.shape[1])

            uv_image_color = cv2.resize(uv_image_color, (width, height))
            red_ir_image_color = cv2.resize(red_ir_image_color, (width, height))

            # Extract the blue channel from the UV image and the red channel from the red/IR image
            blue_channel = uv_image_color[:, :, 0]
            red_channel = red_ir_image_color[:, :, 2]

            # Create a pseudocolor image by combining the extracted channels
            pseudocolor_image = np.zeros((height, width, 3), dtype=np.uint8)
            pseudocolor_image[:, :, 0] = blue_channel  # Blue channel
            pseudocolor_image[:, :, 2] = red_channel   # Red channel

            # Convert the pseudocolor image to PIL format for display in Tkinter
            pseudocolor_image_pil = Image.fromarray(pseudocolor_image)

            # Resize the pseudocolor image to fit within the main window
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

    label_image1 = ctk.CTkLabel(frame_inputs, text="Image 1:")
    label_image1.grid(row=0, column=0, pady=5, padx=5, sticky="w")
    entry_image1 = ctk.CTkEntry(frame_inputs, width=250)
    entry_image1.grid(row=0, column=1, pady=5, padx=5)
    button_browse1 = ctk.CTkButton(frame_inputs, text="Browse", command=lambda: open_file(entry_image1))
    button_browse1.grid(row=0, column=2, pady=5, padx=5)

    label_image2 = ctk.CTkLabel(frame_inputs, text="Image 2:")
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


def power():
    print("Power selected")

def invert():
    print("Invert selected")

def rotateRight():
    print("Rotate Right selected")

def rotateLeft():
    print("Rotate Left selected")

def blurandDivide():
    print("Blur and Divide selected")

def PCA():
    print("PCA selected")

def SAM():
    print("SAM selected")

def save():
    print("Save")

def import_file():
    print("Import")

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
button_effect2 = ctk.CTkButton(buttons_frame, text="sharpie", command=sharpie, width=100, height=40)
button_effect3 = ctk.CTkButton(buttons_frame, text="power", command=power, width=100, height=40)
button_effect4 = ctk.CTkButton(buttons_frame, text="invert", command=invert, width=100, height=40)
button_effect5 = ctk.CTkButton(buttons_frame, text="rotateRight", command=rotateRight, width=100, height=40)
button_effect6 = ctk.CTkButton(buttons_frame, text="rotateLeft", command=rotateLeft, width=100, height=40)
button_effect7 = ctk.CTkButton(buttons_frame, text="blur&Divide", command=blurandDivide, width=100, height=40)
button_effect8 = ctk.CTkButton(buttons_frame, text="PCA", command=PCA, width=100, height=40)
button_effect9 = ctk.CTkButton(buttons_frame, text="SAM", command=SAM, width=100, height=40)

button_effect1.grid(row=0, column=0, padx=5)
button_effect2.grid(row=0, column=1, padx=5)
button_effect3.grid(row=0, column=2, padx=5)
button_effect4.grid(row=0, column=3, padx=5)
button_effect5.grid(row=0, column=4, padx=5)
button_effect6.grid(row=0, column=5, padx=5)
button_effect7.grid(row=0, column=6, padx=5)
button_effect8.grid(row=0, column=7, padx=5)
button_effect9.grid(row=0, column=8, padx=5)

# Create Save and Import buttons
save_import_frame = ctk.CTkFrame(top_frame)
save_import_frame.grid(row=0, column=2, padx=40, pady=10, sticky="ne")

button_save = ctk.CTkButton(save_import_frame, text="Save", command=save, width=60, height=30)
button_import = ctk.CTkButton(save_import_frame, text="Import", command=import_file, width=60, height=30)

button_save.grid(row=0, column=0, padx=5)
button_import.grid(row=0, column=1, padx=5)

# Adjust column weights for resizing
top_frame.grid_columnconfigure(0, weight=0)
top_frame.grid_columnconfigure(1, weight=1)
top_frame.grid_columnconfigure(2, weight=0)

# Create a label for the welcome message
label = ctk.CTkLabel(root, text="Welcome to AppName, Choose an effect to get started", font=("Helvetica", 16))
label.pack(pady=20)

# Create a label to display the result image
result_label = ctk.CTkLabel(root, text="")
result_label.pack(pady=20)

# Run the main loop
root.mainloop()
