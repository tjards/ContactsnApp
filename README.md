<div style="text-align: center;">
  <img src="./reports/CsnApp.jpeg" width="30%" style="margin-bottom: 1em;" />
  <p>
    Full description available <a href="https://tjards.github.io/posts/2025/contactsnapp/">here</a>.
  </p>
</div>

<!---
# ContactsnApp: find phone numbers in images

Building off my earlier exploration of [custom ChatGPTs](https://tjards.github.io/posts/2024/custom_gpt/), I tried building an AI app from scratch based most off vibe coding supported by [GitHub Copilot in VSCode](https://code.visualstudio.com/docs/copilot/overview). Inspired by a recent AirBnB experience, I developed ContactsnApp: a user-friendly application designed to streamline the extraction and annotation of phone numbers from images using AI models. 

The first thing I asked ChatGPT to do was recommend which tools I should use, which I summarize as follows:
- **Frontend**: [Toga](https://beeware.org/project/toga/), a Python native, OS native GUI toolkit. 
- **Backend**: [Flask](https://flask.palletsprojects.com/en/stable/), a lightweight web application framework to host my AI models
- **Object detection and classification**: [YOLO](https://www.v7labs.com/blog/yolo-object-detection), a popular object detection model known for its speed and accuracy
- **Computer vision dataset**: [Roboflow](https://roboflow.com/), to train my YOLO model
- **Integration**: frontend and backend components communicate via [REST API](https://restfulapi.net/).

## Key Functionalities
- **Image Upload**: Users select and upload images directly within the app.
- **Digit Detection**: Uses YOLO to detect digits as objects (bounded boxes).
- **Classification**: Uses Convolutional Neural Network (CNN)-based classification to detect and label digits within these bounded boxes.
- **Post-processing**: Some custom post-processing techniques to clean up the results.
- **Interactive Display**: Provides annotated images displaying detected numbers clearly.


<img src="reports/GUI.png" width="70%"/>

## Project Structure

Here's an outline of the application's structure. 

```
contactsnap/
├── backend/
│   ├── app.py                  # Flask backend API
│   ├── inference/
│   │   ├── detect_classify.py  # YOLO/CNN detection and classification pipeline
│   │   └── models/
│   │       └── [model files]   # Model weights go here (nominally in *.pt format)
│
├── frontend/
│   ├── app.py                  # Toga GUI application
│   ├── uploads/                # uploaded images are saved here
│   └── results/                # annotated images are saved here
└── app.py                      # Launches frontend and backend
```

## Learning process

I also experimented with transfer learning. I first trained my YOLO model to identify cleanish characters as objects using [alphabettraining-iis85](https://universe.roboflow.com/alphabettraining/character-detection-iis85) and then fine-tuned for handwritten digits using [cscai-o5oyu](https://universe.roboflow.com/cscai-o5oyu/handwriting-recognition-xyekz). I "froze" the first 10 layers of the YOLO model in order to speed up learning, assuming that the first run had learned the basic structure of digit objects and only needed fine-tuning to recognize the variance in handwritten digits.

### Learning the structure of digits

Below are training and validation progress for the first training run, where I trainined the whole model on digit objects. Note that the training and validation loss both decline. At the end, validation loss is just above 1.

<img src="reports/learning_results_a.png" width="50%"/>

### Finetuning for handwritten digits

Below are training and validation progress for the second training run, where I froze the first 10 layers of the model and focused mainly on improving the output layer. Note that the training and validation loss both decline lower than the first run. At the end, validation loss is around 0.5.

<img src="reports/learning_results_b.png" width="50%"/>

## Post-processing

In order to filter out non-phone numbers, I develop a custom post-processing technique that only returns numbers that are grouped close together horizontally.

Below you will see this works to filter out some letters mis-identified as part of the phone number:

<img src="reports/output_with_labels_nopost.jpg" width="30%"/>
<img src="reports/output_with_labels_post.jpg" width="30%"/>

## Building the App with Briefcase

I used Briefcase to build the app. If you want to run the app yourself, make sure Python and Briefcase are installed:
```bash
pip install briefcase
```
Create the project:
```bash
briefcase create
```
Build the app:
```bash
briefcase build
```
Run the app:
```bash
briefcase run
```

I don't pay the $99/mo to be an Apple developer, so I could not bring to distribution. If you are so-inclined, this will package for distribution:
```bash
briefcase package
```

## Future Improvements
- Notice the model make mistakes, especially with handwritten digits and poor lighting. I plan to improve the models. 
- Make a mobile app.
- Autosave to mobile contact list.

--->

