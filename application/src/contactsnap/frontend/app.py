# front end for contactsnap app

# ------------------------------
# IMPORT STUFF
# ------------------------------

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, CENTER
import requests
import os
from io import BytesIO
from PIL import Image

# -----------------------------------
# CUSTOM APP CLASS
# ----------------------------------- 

class ContactSnap(toga.App):

    BASE_URL = "http://127.0.0.1:5050"

    def startup(self):

        # create the main interface box
        main_box = toga.Box(style=Pack(direction=COLUMN, alignment=CENTER))

        # label to show status or results
        self.label = toga.Label("Upload a photo to detect phone numbers", style=Pack(padding=10))
        # placeholder
        self.image_view = toga.ImageView(style=Pack(height=300, padding=10))
        # [upload button], links to "upload_image" function below
        self.upload_button = toga.Button("Choose Image", on_press=self.upload_image, style=Pack(padding=10))
        # [detection button]
        self.detect_button = toga.Button("Detect Phone Number", on_press=self.detect_phone_number, style=Pack(padding=10))
        # disabled at first, until an image is uploaded
        self.detect_button.enabled = False

        # add above widgets to main box
        main_box.add(self.label)
        main_box.add(self.image_view)
        main_box.add(self.upload_button)
        main_box.add(self.detect_button)

        # set up window
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    # upload the image
    async def upload_image(self, widget):
        
        # file selection
        file_path = await self.main_window.dialog(toga.OpenFileDialog("Choose an image file"))
        
        # if a file was selected
        if file_path:

            # save the path to the image
            self.image_path = file_path

            # display the selected image
            self.image_view.image = toga.Image(file_path)
            self.label.text = "Image selected. Ready to detect."
            
            # enable the detection button
            self.detect_button.enabled = True

    # detect and classify the phone number
    def detect_phone_number(self, widget):

        #BASE_URL = "http://localhost:5050"
        #BASE_URL = "http://127.0.0.1:5050"
        self.label.text = "Uploading and detecting..."
        LOCAL_SAVE = False # set to True to save the image locally (for later mobile app)

        try:

            # Ping backend first
            ping_response = requests.get(f"{self.BASE_URL}/ping", timeout=2)
            if ping_response.status_code != 200 or ping_response.text != "pong":
                self.label.text = "Backend is not available. Please restart the app."
                return

            # upload the image
            with open(self.image_path, 'rb') as f:
                upload_response = requests.post(f"{self.BASE_URL}/upload", files={'image': f})

            # if the upload is successful (i.e., status code 200)
            if upload_response.status_code == 200:
                self.label.text = upload_response.text

            else:
                self.label.text = f"Upload failed: {upload_response.status_code}"
                return

            # request detection service
            detect_response = requests.get(f"{self.BASE_URL}/detect")

            # if the detection is successful (i.e., status code 200)
            if detect_response.status_code == 200:

                # pull the image from the response
                img = Image.open(BytesIO(detect_response.content))
                
                # display the image in the app
                self.image_view.image = toga.Image(img)

                # if we want to save the image locally (may be useful later)
                if LOCAL_SAVE:
                    img_path = os.path.join(os.path.expanduser("~"), "annotated_image.jpg")
                    img.save(img_path)

                self.label.text = "Detection and Classification complete."

            else:

                self.label.text = f"Detection failed: {detect_response.status_code}"

        except Exception as e:
            
            self.label.text = f"Error: {e}"

def main():
    return ContactSnap()
