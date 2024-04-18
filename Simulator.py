from cluster import Cluster
from Automated_Annotator import Automated_Annotator
import Train_YOLOv8
import Edit_Detector
from torch import nn
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import argparse
import cv2
from YOLOv8 import YOLOv8
from shapely.geometry import box
from shapely.ops import unary_union
from collections import defaultdict
import ast


def select_directory():
    # Create Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open file dialog to select directory
    directory = filedialog.askdirectory(initialdir='C:/Users', title="Select Directory")

    # Check if directory is selected
    if directory:
        print("Selected Directory:", directory)
        return directory
    else:
        print("No directory selected.")
        return None


class Simulator:
    def __init__(self, image_directory, model_name):
        self.image_directory = image_directory
        self.cluster = Cluster()
        # # self.annotator = Automated_Annotator()
        self.modelDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
        self.folder_name = os.path.basename(self.image_directory)


        # Check if the directory exists
        if not os.path.exists(self.modelDir):
            # If the directory doesn't exist, create it
            os.makedirs(self.modelDir)
        #
        # # Prompt the user to create another folder inside the Models directory
        # new_folder_name = input("Please enter a name for the model: ")
        new_folder_name = model_name
        self.new_folder_path = os.path.join(self.modelDir, new_folder_name)
        self.train_path = os.path.join(self.new_folder_path, self.folder_name)
        # Check if the new directory exists
        if not os.path.exists(self.new_folder_path):
            # If the directory doesn't exist, create it
            os.makedirs(self.new_folder_path)
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)

        self.label_file = pd.read_csv(os.path.join(self.image_directory, f"{self.folder_name}_Labels.csv"))
        self.filenames = [self.label_file["FileName"][i] for i in self.label_file.index]


    def run_clustering(self):
        self.image_names = self.cluster.getBestImages(self.filenames, self.image_directory)
        return self.image_names

    def run_same_interval(self):
        # Calculate the interval
        interval = len(self.label_file) // len(self.image_names)

        # Generate indices at equal intervals
        indices = range(0, len(self.label_file), interval)

        # If len(indices) is not equal to len(self.image_names), remove the last indices
        while len(indices) > len(self.image_names):
            indices = indices[:-1]

        # Select the rows at these indices from self.label_file
        selected_rows = self.label_file.iloc[indices]

        # Extract the 'FileName' column from the selected rows and update self.image_names
        self.image_names = selected_rows['FileName'].tolist()
        return self.image_names

    def run_keyframes(self):
        interval = 10

        # Generate indices at every 10th row
        indices = range(0, len(self.label_file), interval)

        # Select the rows at these indices from self.label_file
        selected_rows = self.label_file.iloc[indices]

        # Extract the 'FileName' column from the selected rows and update self.image_names
        self.image_names = selected_rows['FileName'].tolist()
        return self.image_names

    def createWorkingCSV(self, method):
        # Create a new DataFrame with the same number of rows as self.label_file
        self.workingCSV = pd.DataFrame(index=self.label_file.index)

        # Set the "Folder" column to self.image_directory
        self.workingCSV["Folder"] = self.image_directory

        # Copy the "FileName" column from self.label_file to the new DataFrame
        self.workingCSV["FileName"] = self.label_file["FileName"]

        # Set the "Status" column to "Incomplete"
        self.workingCSV["Status"] = "Incomplete"

        # Set the "Bounding boxes" column to "[]"
        self.workingCSV["Bounding boxes"] = "[]"

        # For the rows where "FileName" is in self.image_names, update the "Status" to "Complete" and "Bounding boxes" to the corresponding value from self.label_file's "Tool bounding box" column
        for i in self.workingCSV.index:
            if self.workingCSV.loc[i, "FileName"] in self.image_names:
                self.workingCSV.loc[i, "Status"] = "Complete"
                self.workingCSV.loc[i, "Bounding boxes"] = self.label_file.loc[i, "Tool bounding box"]

        # Save the DataFrame to a CSV file
        self.workingCSV.to_csv(os.path.join(self.train_path, f"WorkingCSV_{method}.csv"), index=False)

    def createTrainingCSV(self, method):
        # self.workingCSV.to_csv(os.path.join(self.train_path, f"WorkingCSV_{method}.csv"), index=False)
        entries = self.workingCSV.loc[self.workingCSV["Status"] == "Complete"]
        trainCSV = pd.concat([entries.copy(), entries.copy(), entries.copy()])
        trainCSV.index = [i for i in range(3 * len(entries.index))]
        trainCSV["Fold"] = [0 for i in range(3 * len(entries.index))]
        set_names = ["Train" for i in range(len(entries.index))] + ["Validation" for i in
                                                                    range(len(entries.index))] + ["Test" for i in
                                                                                                  range(
                                                                                                      len(entries.index))]
        trainCSV["Set"] = set_names
        self.save_path = os.path.join(self.train_path, f"Train_data_{method}.csv")
        trainCSV.to_csv(os.path.join(self.save_path), index=False)

    def createautoTrainingCSV(self, image_files, method):
        self.workingCSV= pd.read_csv(os.path.join(self.train_path, f"WorkingCSV_{method}.csv"))
        entries = self.workingCSV.loc[self.workingCSV["FileName"].isin(image_files)]
        trainCSV = pd.concat([entries.copy(), entries.copy(), entries.copy()])
        trainCSV.index = [i for i in range(3 * len(entries.index))]
        trainCSV["Fold"] = [0 for i in range(3 * len(entries.index))]
        set_names = ["Train" for i in range(len(entries.index))] + ["Validation" for i in
                                                                    range(len(entries.index))] + ["Test" for i in
                                                                                                  range(
                                                                                                      len(entries.index))]
        trainCSV["Set"] = set_names
        self.save_path = os.path.join(self.train_path, f"Train_data_{method}.csv")
        trainCSV.to_csv(os.path.join(self.save_path), index=False)

    def train_YOLOv8(self):
        parser = argparse.ArgumentParser('YOLOv8 training script', parents=[Train_YOLOv8.get_arguments()])
        args = parser.parse_args()
        args.save_location = os.path.join(self.train_path)
        args.data_csv_file = os.path.join(self.save_path)
        if len(pd.read_csv(self.save_path).index)>1000:
            balance = False
        else:
            balance=True
        if os.path.exists(os.path.join(self.train_path,"train")):
            epochs = 2
        else:
            epochs = 100
        args.balance = balance
        args.epochs = epochs
        Train_YOLOv8.train(args)

    def predict_all_frames(self, method):
        # Load the trained YOLO model
        self.yolo = YOLOv8("detect")
        self.yolo.loadModel(os.path.join(self.train_path))

        # Iterate over the rows of the workingCSV DataFrame
        for i in self.workingCSV.index:
            # Check if the status is "Incomplete"
            if self.workingCSV.loc[i, "Status"] == "Incomplete":
                # Load the corresponding image
                image_path = os.path.join(self.workingCSV.loc[i, "Folder"], self.workingCSV.loc[i, "FileName"])
                image = cv2.imread(image_path)

                # Use the YOLO model to predict the bounding boxes for the image
                predicted_bboxes = self.yolo.predict(image)

                # Update the "Bounding boxes" column with the predicted bounding boxes
                self.workingCSV.loc[i, "Bounding boxes"] = predicted_bboxes

                # Change the "Status" column to "Review"
                self.workingCSV.loc[i, "Status"] = "Review"

        # Save the updated workingCSV DataFrame to a CSV file
        self.workingCSV.to_csv(os.path.join(self.train_path, f"WorkingCSV_{method}.csv"), index=False)

    def predict_next_set(self, data, method):
        # Load the trained YOLO model
        self.yolo = YOLOv8("detect")
        self.yolo.loadModel(os.path.join(self.train_path))

        # Get the next set of images
        next_set_images = data

        # Iterate over the rows of the workingCSV DataFrame
        for i in self.workingCSV.index:
            # Check if the image is in the next set
            if self.workingCSV.loc[i, "FileName"] in next_set_images:
                # Load the corresponding image
                image_path = os.path.join(self.workingCSV.loc[i, "Folder"], self.workingCSV.loc[i, "FileName"])
                image = cv2.imread(image_path)

                # Use the YOLO model to predict the bounding boxes for the image
                predicted_bboxes = self.yolo.predict(image)

                # Update the "Bounding boxes" column with the predicted bounding boxes
                self.workingCSV.loc[i, "Bounding boxes"] = predicted_bboxes

                # Change the "Status" column to "Review"
                self.workingCSV.loc[i, "Status"] = "Review"

        # Save the updated workingCSV DataFrame to a CSV file
        self.workingCSV.to_csv(os.path.join(self.train_path, f"WorkingCSV_{method}.csv"), index=False)

    def get_next_set(self, review, method):
        # Get the indices of the rows where the status is "Complete"
        self.workingCSV = pd.read_csv(os.path.join(self.train_path, f"WorkingCSV_{method}.csv"))
        next_set_rows = []
        if not review:
            complete_indices = self.workingCSV[self.workingCSV["Status"] == "Complete"].index
            for i in complete_indices:
                if i - 1 >= 0 and i + 1 < len(self.workingCSV):
                    next_set_rows.extend([i - 1, i + 1])

        elif review:
            review_indices = self.workingCSV[self.workingCSV["Status"] == "Reviewed"].index
            for i in review_indices:
                if self.workingCSV["Status"][i] == "Reviewed":
                    if i - 1 >= 0 and i + 1 < len(self.workingCSV):
                        if self.workingCSV.loc[i - 1, "Status"] == "Incomplete":
                            next_set_rows.append(i - 1)
                        if self.workingCSV.loc[i + 1, "Status"] == "Incomplete":
                            next_set_rows.append(i + 1)

        next_set_df = self.workingCSV.loc[next_set_rows]
        file_names = list(set(next_set_df['FileName'].tolist()))

        return file_names

    def selection_by_IOU(self, review):
        next_set_rows = []
        if review:
            indicator = "Reviewed"
        else:
            indicator = "Complete"


        review_indices = self.workingCSV[self.workingCSV["Status"] == indicator].index
        for i in review_indices:
            if self.workingCSV["Status"][i] == indicator:
                if i - 1 >= 0 and i + 1 < len(self.workingCSV):
                    if self.workingCSV.loc[i - 1, "Status"] == "Review":
                        referenceboxes = ast.literal_eval(self.workingCSV.loc[i, "Bounding boxes"])
                        predboxes = ast.literal_eval(self.workingCSV.loc[i - 1, "Bounding boxes"])
                        if self.find_true_matches(referenceboxes, predboxes, 0.8):
                            self.workingCSV.loc[i - 1, "Status"] = "Reviewed"
                        else:
                            next_set_rows.append(i - 1)
                    if self.workingCSV.loc[i + 1, "Status"] == "Review":
                        referenceboxes = ast.literal_eval(self.workingCSV.loc[i, "Bounding boxes"])
                        predboxes = ast.literal_eval(self.workingCSV.loc[i + 1, "Bounding boxes"])
                        if self.find_true_matches(referenceboxes, predboxes, 0.8):
                            self.workingCSV.loc[i + 1, "Status"] = "Reviewed"
                        else:
                            next_set_rows.append(i + 1)
        print("next set rows", next_set_rows)
        self.workingCSV.to_csv(os.path.join(self.train_path, "WorkingCSV_clustering.csv"))
        next_set_df = self.workingCSV.loc[next_set_rows]
        file_names = list(set(next_set_df['FileName'].tolist()))
        print(file_names)

        return file_names

    def calculate_iou(self,bbox1, bbox2):
        box1 = box(bbox1['xmin'], bbox1['ymin'], bbox1['xmax'], bbox1['ymax'])
        box2 = box(bbox2['xmin'], bbox2['ymin'], bbox2['xmax'], bbox2['ymax'])
        intersection_area = box1.intersection(box2).area
        union_area = unary_union([box1, box2]).area
        iou = intersection_area / union_area
        return iou

    def find_true_matches(self, rboxes, pboxes, iou_threshold):
        true_boxes = rboxes.copy()
        predicted_boxes = pboxes.copy()
        for predicted_bbox in predicted_boxes.copy():  # Iterate over a copy of predicted_boxes
            if true_boxes:
                ious = [self.calculate_iou(predicted_bbox, truth_bbox) for truth_bbox in true_boxes]
                max_iou_index = ious.index(max(ious))

                if ious[max_iou_index] >= iou_threshold:
                    if predicted_bbox['class'] == true_boxes[max_iou_index]['class']:
                        true_boxes.pop(max_iou_index)
                        predicted_boxes.remove(predicted_bbox)

        # If all boxes have been matched, return True. Otherwise, return False.
        return len(true_boxes) == 0 and len(predicted_boxes) == 0

    def prep_data(self, method):
        self.workingCSV = pd.read_csv(os.path.join(self.modelDir, "CataractSurgery", f"WorkingCSV_{method}.csv"))

        # Load the ground truth and predicted data
        truth_data = self.label_file
        predicted_data = self.workingCSV

        # Merge the ground truth and predicted data on 'FileName'
        self.merged_data = pd.merge(predicted_data, truth_data, on='FileName')
        self.complete_true_data, self.complete_pred_data, self.review_true_data, self.review_pred_data, self.next_two_truth_data, self.next_two_predicted_data = Edit_Detector.get_data(self.merged_data)

    def compare_ground_truth(self, method):
        # Create a list of thresholds from 0.5 to 0.9 with an interval of 0.05
        thresholds = [i / 100 for i in range(50, 95, 5)]

        # Initialize a dictionary to store the results
        results = {}

        # For each threshold, count the bounding boxes to edit
        for threshold in thresholds:
            complete_bboxes_to_edit = Edit_Detector.calculate_counts(self.complete_true_data, self.complete_pred_data,
                                                                     threshold)

            incomplete_bboxes_to_edit = Edit_Detector.calculate_counts(self.review_true_data, self.review_pred_data,
                                                                       threshold)

            next_two_bboxes_to_edit = Edit_Detector.calculate_counts(self.next_two_truth_data,
                                                                     self.next_two_predicted_data,
                                                                     threshold)

            # Store the results in the dictionary
            results[threshold] = {
                'complete': complete_bboxes_to_edit,
                'incomplete': incomplete_bboxes_to_edit,
                'next_two': next_two_bboxes_to_edit
            }

            # Create a DataFrame to store the results
            results_df = pd.DataFrame(
                columns=['Total Images', 'Complete Images', 'Review Images',
                         # 'Next Two Rows Images',
                         'Total True BBoxes Complete', 'Total Predicted BBoxes Complete', 'Total True BBoxes Review',
                         'Total Predicted BBoxes Review',
                         # 'Total True BBoxes Next Two Rows',
                         # 'Total Predicted BBoxes Next Two Rows',
                         # 'Total Edits Next Two Rows 0.5 IOU',
                         # 'Total Edits Next Two Rows 0.9 IOU',
                         # 'Total Edits Next Two Rows Average',
                         'Correct Boxes Complete',
                         'Total Edits Review 0.5 IOU',
                         'Additions Review 0.5 IOU', 'Deletions Review 0.5 IOU',
                         'Renames Review 0.5 IOU', 'Reposition & Resizing Review 0.5 IOU',
                         'Correct Boxes Review 0.5 IOU',
                         'Total Edits Review 0.9 IOU',
                         'Additions Review 0.9 IOU', 'Deletions Review 0.9 IOU', 'Renames Review 0.9 IOU',
                         'Reposition & Resizing Review 0.9 IOU',
                         'Correct Boxes Review 0.9 IOU',
                         'Total Edits Review Average',
                         'Additions Review Average', 'Deletions Review Average', 'Renames Review Average',
                         'Reposition & Resizing Review Average',
                         'Correct Boxes Review Average'])

            # Append the results to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([{
                'Total Images': len(self.workingCSV),
                'Complete Images': len(self.complete_true_data),
                'Review Images': len(self.review_true_data),
                # 'Next Two Rows Images': len(self.next_two_predicted_data) + len(self.next_two_truth_data),
                'Total True BBoxes Complete': Edit_Detector.count_total_bboxes(self.complete_true_data),
                'Total Predicted BBoxes Complete': Edit_Detector.count_total_bboxes(self.complete_pred_data),
                'Total True BBoxes Review': Edit_Detector.count_total_bboxes(self.review_true_data),
                'Total Predicted BBoxes Review': Edit_Detector.count_total_bboxes(self.review_pred_data),
                # 'Total True BBoxes Next Two Rows': Edit_Detector.count_total_bboxes(self.next_two_truth_data),
                # 'Total Predicted BBoxes Next Two Rows': Edit_Detector.count_total_bboxes(self.next_two_predicted_data),
                # 'Total Edits Next Two Rows 0.5 IOU': results[0.5]["next_two"][4],
                # 'Total Edits Next Two Rows 0.9 IOU': results[0.9]["next_two"][4],
                # 'Total Edits Next Two Rows Average': round(
                #     sum([results[t]["next_two"][4] for t in thresholds]) / len(thresholds)),
                'Correct Boxes Complete': complete_bboxes_to_edit[5],
                'Total Edits Review 0.5 IOU': results[0.5]["incomplete"][4],
                'Additions Review 0.5 IOU': results[0.5]["incomplete"][0],
                'Deletions Review 0.5 IOU': results[0.5]["incomplete"][1],
                'Renames Review 0.5 IOU': results[0.5]["incomplete"][2],
                'Reposition & Resizing Review 0.5 IOU': results[0.5]["incomplete"][3],
                'Correct Boxes Review 0.5 IOU': results[0.5]["incomplete"][5],
                'Total Edits Review 0.9 IOU': results[0.9]["incomplete"][4],
                'Additions Review 0.9 IOU': results[0.9]["incomplete"][0],
                'Deletions Review 0.9 IOU': results[0.9]["incomplete"][1],
                'Renames Review 0.9 IOU': results[0.9]["incomplete"][2],
                'Reposition & Resizing Review 0.9 IOU': results[0.9]["incomplete"][3],
                'Correct Boxes Review 0.9 IOU': results[0.9]["incomplete"][5],
                'Total Edits Review Average': round(
                    sum([results[t]["incomplete"][4] for t in thresholds]) / len(thresholds)),
                'Additions Review Average': round(
                    sum([results[t]["incomplete"][0] for t in thresholds]) / len(thresholds)),
                'Deletions Review Average': round(
                    sum([results[t]["incomplete"][1] for t in thresholds]) / len(thresholds)),
                'Renames Review Average': round(
                    sum([results[t]["incomplete"][2] for t in thresholds]) / len(thresholds)),
                'Reposition & Resizing Review Average': round(
                    sum([results[t]["incomplete"][3] for t in thresholds]) / len(thresholds)),
                'Correct Boxes Review Average': round(sum([results[t]["incomplete"][5] for t in thresholds]) / len(thresholds))
            }])], ignore_index=True)
            # Save the DataFrame to a CSV file
            results_df.to_csv(os.path.join(self.modelDir, "CataractSurgery", f"comparison_results_{method}.csv"),
                              index=False)


        # Print the results to a text file
        with open(os.path.join(self.modelDir, "CataractSurgery", f"comparison_results_{method}.txt"), 'w') as f:
            f.write(f'Total number of Images: {len(self.workingCSV)}\n')
            f.write(f'Number of Images with status "Complete": {len(self.complete_true_data)}\n')
            f.write(f'Number of Images with status "Review": {len(self.review_true_data)}\n')
            f.write(
                f'Number of Images for the next two rows after a "Complete" row: {len(self.next_two_predicted_data) + len(self.next_two_truth_data)}\n')

            f.write(f'\n -------------------------------Total-------------------------------------------- \n')
            f.write(
                f'Total number of true bounding boxes / instances for complete data: {Edit_Detector.count_total_bboxes(self.complete_true_data)}\n')
            f.write(
                f'Total number of predicted bounding boxes / instances for complete data: {Edit_Detector.count_total_bboxes(self.complete_pred_data)}\n')
            f.write(
                f'Total number of true bounding boxes / instances for review data: {Edit_Detector.count_total_bboxes(self.review_true_data)}\n')
            f.write(
                f'Total number of predicted bounding boxes / instances for review data: {Edit_Detector.count_total_bboxes(self.review_pred_data)}\n')

            f.write(f'\n ----------------------------- Next 2 frames------------------------------------- \n')
            f.write(
                f'Total number of true bounding boxes / instances for next two rows data: {Edit_Detector.count_total_bboxes(self.next_two_truth_data)}\n')
            f.write(
                f'Total number of predicted bounding boxes / instances for next two rows data: {Edit_Detector.count_total_bboxes(self.next_two_predicted_data)}\n')
            f.write(
                f'Total Number of edits for the next two rows after a "Complete" row - 0.5 IOU: {results[0.5]["next_two"][4]}\n')
            f.write(
                f'Total Number of edits for the next two rows after a "Complete" row - 0.9 IOU: {results[0.9]["next_two"][4]}\n')
            f.write(
                f'Total Number of edits for the next two rows after a "Complete" row - Average: {round(sum([results[t]["next_two"][4] for t in thresholds]) / len(thresholds))}\n')

            # Print the results at thresholds 0.5, 0.9 and the average of all thresholds
            f.write(f'\n -------------------Complete and Remaining Frame Edits------------------------- \n')
            f.write(f'\nNegative test \n')
            f.write(f'Total Number of edits for complete data: {complete_bboxes_to_edit[4]}\n')
            f.write(f'Number of addition for complete data: {complete_bboxes_to_edit[0]}\n')
            f.write(f'Number of deletion for complete data: {complete_bboxes_to_edit[1]}\n')
            f.write(f'Number of renaming for complete data: {complete_bboxes_to_edit[2]}\n')
            f.write(f'Number of reposition & resizing for complete data: {complete_bboxes_to_edit[3]}\n')
            f.write(
                f'Number of correct boxes vs. total predicted boxes: {complete_bboxes_to_edit[5]} / {Edit_Detector.count_total_bboxes(self.complete_pred_data)}\n')

            f.write(f'\nThreshold: 0.5\n')
            f.write(f'Total Number of edits for review data: {results[0.5]["incomplete"][4]}\n')
            f.write(f'Number of addition for review data: {results[0.5]["incomplete"][0]}\n')
            f.write(f'Number of deletion for review data: {results[0.5]["incomplete"][1]}\n')
            f.write(f'Number of renaming for review data: {results[0.5]["incomplete"][2]}\n')
            f.write(f'Number of reposition & resizing for complete data: {results[0.5]["incomplete"][3]}\n')
            f.write(
                f'Number of correct boxes vs. total predicted boxes: {results[0.5]["incomplete"][5]} / {Edit_Detector.count_total_bboxes(self.review_pred_data)}\n')

            f.write(f'\nThreshold: 0.9\n')
            f.write(f'Total Number of edits for review data: {results[0.9]["incomplete"][4]}\n')
            f.write(f'Number of addition for review data: {results[0.9]["incomplete"][0]}\n')
            f.write(f'Number of deletion for review data: {results[0.9]["incomplete"][1]}\n')
            f.write(f'Number of renaming for review data: {results[0.9]["incomplete"][2]}\n')
            f.write(f'Number of reposition & resizing for complete data: {results[0.9]["incomplete"][3]}\n')
            f.write(
                f'Number of correct boxes vs. total predicted boxes: {results[0.9]["incomplete"][5]} / {Edit_Detector.count_total_bboxes(self.review_pred_data)}\n')

            f.write(f'\nAverage\n')
            f.write(
                f'Total Number of edits for review data: {round(sum([results[t]["incomplete"][4] for t in thresholds]) / len(thresholds))}\n')
            f.write(
                f'Number of addition for review data: {round(sum([results[t]["incomplete"][0] for t in thresholds]) / len(thresholds))}\n')
            f.write(
                f'Number of deletion for review data: {round(sum([results[t]["incomplete"][1] for t in thresholds]) / len(thresholds))}\n')
            f.write(
                f'Number of renaming for review data: {round(sum([results[t]["incomplete"][2] for t in thresholds]) / len(thresholds))}\n')
            f.write(
                f'Number of reposition & resizing for review data: {round(sum([results[t]["incomplete"][3] for t in thresholds]) / len(thresholds))}\n')
            f.write(
                f'Number of correct boxes vs. total predicted boxes: {round(sum([results[t]["incomplete"][5] for t in thresholds]) / len(thresholds))} / {Edit_Detector.count_total_bboxes(self.review_pred_data)}\n')
    def run_process(self, run_step, method):
        run_step()
        print("step 1 - running step yes")
        self.createWorkingCSV(method)
        print("step 1.25 - create image_label file yes")
        self.createTrainingCSV(method)
        print("step 1.5 - create training file yes")
        self.train_YOLOv8()
        print("step 2 - training YOLO yes")
        self.predict_all_frames(method)
        print("step 3 - YOLO predicting yes")
        self.compare_ground_truth(method)
        print("step 4 - Evaluating yes")

    def run(self):
        self.run_process(self.run_clustering, 'clustering')
        self.run_process(self.run_same_interval, 'interval')
        self.run_process(self.run_keyframes, 'ten_frame')


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    image_directory = select_directory()
    simulator = Simulator(image_directory)
    simulator.run()