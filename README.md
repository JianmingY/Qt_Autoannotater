# Qt_Autoannotater Instructions:

1. Required packages:
   - pytorch
   - PyQt5
   - pandas
   - PIL
   - numpy
   - random
   - os
   - sys
   - cv2
   - yaml
   - torchvision
   - ultralytics
   - transformers
   - copy
   - gc
   - sklearn
   - tkinter
   - argparse
   - shapely
   - ast
   - sys
   - maybe more let me know
   - 
2. Prepare needed files:
   - A folder of images
   - Label file
   - *** prepare a txt file with each line being a class of tools/objects ***

3. Once running:
   - click Rregular Annotate' for mannual annotation and 'Auto Annotate' for semi-automated annotation
   - click 'load label classes' to select the class txt file created in step 2
   - click 'open file' to select the folder where the images are stored
   - click 'OK' when it says the key frame selection is initiated and *wait*
   - mannualy annotate the selected images after finishing one image click 'D' to save labels
   - after all selected images are annotated click 'auto annotate'
   - click 'yes' if you have correctly annotated these images and ok to *wait* for model training and predicting
   - the programm will show the next set of selected images with predicted bouding boxed on them, correct them then click 'D' to complete the reviewing for each image
   - once the set is reviewed, click auto annotate again, it will run less training epochs from here
   - once the all the images are labeled and reviewed, clicking on 'auto annotate' will tell you auto annotate is completed and show you all images
   - You Label file should already be updated, you can try visualizer to see if this is actually correct
  
4. Background:
   - ***wait*** means that the window will freeze, please try not to do anything at that moment
   - the labels are saved in YOLO txt file format for all labeled images in the image folder and they are also updated in your Label file
          # Additionally, while annotation the working process and results can be observed in the pulled Qt_Autoannotater folder's Model folder
   - The program enables Manual select number of frames to annotate by checking that box, this is because the auto selection will become very frequent and only a small number of images will be selected for editing, ***not recommended*** but you can turn it on to accelerated the annotation process when confident and needed
       
# Annotation commands

1. Adding bounding box: click key 1~9 to select your object class in the same order of the class txt file you created then left click to drag and drop
2. Deleting bounding box: hover you pointer on a box and click 'E' or 'Backspace'
3. Adjusting bounding box (not recommended): hover pointed on a box and click 'A' drage the squares shown to adjust box
4. Moving bounding box: right click and hold a box to move to where ever you want
5. Clear all labels: click 'C' to remove all boxes on the current image
6. Copy and paste from the previous image (not recommended): click 'V'
7. Go to next image and save: click 'D'
8. Go to previous image without saving: click 'S'
9. Go to 10 imgae later and save (not recommended): click 'F'
