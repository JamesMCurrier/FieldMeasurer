# Field Measurer
### Before
![Before Image](https://github.com/JamesMCurrier/FieldMeasurer/imgs_for_readme/before.jpg)

### After
![After Image](https://github.com/JamesMCurrier/FieldMeasurer/imgs_for_readme/after.jpg)

## Problem

Planting accuracy of potato fields is extremely important as plants that are too close together will have a smaller yield, and plants that are too far apart are a waste of space. Currently, the distances between plants are being measured manually by hand on a ten-foot strip of each field. This method is slow, prone to human error, and the small sample size does not adequately represent the planting of the entire field. This project aims to provide an extremely fast, accurate way to measure distances between plants in a field to ensure proper planting.


## Solution

To solve this program, a drone was deployed and set on a path to take thousands of photos of multiple potato fields. The timing of our sample collection was important, as the plants had to be at the correct stage of growth. If the growth was too little then the plants would not be visible from the drone, but if the growth was too much then the plants would be overlapping, and it would be difficult to distinguish individual plants. These photos were then processed by this program, which outputted the analyzed images, and csv files containing additional data about each image.


## How To Run It
### Prerequisites
`Python 3.5` or later
`PIL (Python Imaging Library)` and its dependencies - [PIL website](https://pillow.readthedocs.io/en/stable/index.html)
`sklearn` OPTIONAL - uses neural network to increase speed and accuracy

### Installing
Simply clone or download the repository
```git clone https://github.com/JamesMCurrier/FieldMeasurer.git```
or [click here to download](https://github.com/JamesMCurrier/FieldMeasurer/archive/master.zip)

### Running The Tests
#### Running on a single image
Before starting, go to the "Samples" folder and open "Sample1.jpg". This is the photo we will analyse in this test. 
Run Potatoes.py with Python.
When prompted "Enter Photo Name or Folder Name:" input `Samples/Sample1.jpg` and press enter.
When Prompted "Which layer to display?" input `default` and press enter.
The processed image should be displayed. Zoom in and take a closer look!
You will notice that a new folder called "data" has been created. This folder contains 2 csv files containing data about the field. "data/Distances.csv" contains all of the gap distances. "data/Rows.csv" contains statistics about each of the rows.
Let's try displaying a different image. Go back to the program, input `digital`, and press enter.
You should see a digitized image displayed. Zoom in and take a closer look!
Let's try displaying a custom image. Go back to the program, input `row_ids, lines, boxes, centers, numbers` and press enter
Our custom image should be displayed. Zoom in and take a closer look!
Here's a quick break down of our input...
+ `row_ids` adds the large light blue row numbers on the left.
+ `lines` adds a yellow line through each row.
+ `boxes` adds the pink bounding box around each plant.
+ `centers` adds a blue circle at the center of each plant
+ `numbers` adds the little white numbers coresponding to each of the gaps

To view all of the possible options, input "help", and press enter.
To exit the program, input `quit` and press enter.


#### Running on folder of images
Before starting, go to the "Samples" folder and look at the sample images. We will perform analysis on all of these images.
Run Potatoes.py with Python.
When prompted "Enter Photo Name or Folder Name:" input `Samples` and press enter.
When prompted "Enter number of output images:", input `3` and press enter.
Run Potatoes.py with Python.
When prompted "Enter options for output image #1:" input `default` and press enter.
When prompted "Enter options for output image #2:" input `digital` and press enter.
When prompted "Enter options for output image #3:" input `row_ids, lines, boxes, centers, numbers` and press enter.
The program will begin processing the images. Depending on your system, this may take a while (no more than a few minutes).
When the program terminates, go to the "Samples" folder. You will notice that a new folder called "Analysis" has been created.
Inside this "Analysis" folder, there is a "Data" folder and an "Images folder". 
+ "Data" contains a "Rows.csv" file and a "Distances.csv" file for each of the images.
+ "Images" 3 processed images for each of the original images; one with `default` settings, one with `digital` settings and one with `row_ids, lines, boxes, centers, numbers` settings.

### Additional Information
For additional infomation about the project including results, see [Field_Measurer.pdf](https://github.com/JamesMCurrier/FieldMeasurer/Field_Measurer.pdf)