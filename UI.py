import customtkinter as ctk
from PIL import Image, ImageTk

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


# Function placeholders for buttons
def pseudocolor():
    print("Pseudocolor selected")


def sharpie():
    print("Sharpie selected")


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
label.pack(pady=200)

# Run the main loop
root.mainloop()
