import cv2
import numpy as np
from utils.args import get_args

args, unknown = get_args()

class BoundingBox:
    def __init__(self, class_id, confidence, x1, x2, y1, y2):
        self.class_id = class_id
        self.confidence = confidence
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def box(self):
        return self.x1, self.y1, self.x2, self.y2


class ObjectDetector:
    def __init__(self,
                 image,
                 session,
                 iou_threshold=args.iou_threshold,
                 confidence_threshold=args.confidence_threshold):

        self.img = image
        self.image_shape = image.shape
        self.session = session
        self.iou = iou_threshold
        self.confidence = confidence_threshold
        self.input_dimensions = session.get_inputs()[0].shape
        self.predictions = None
        self.boxes = None
        self.scores = None
        self.class_ids = None
        self.detected_objects = []

    def preprocess(self):
        """Converts the image to the input format the model expects"""
        input_shape = self.input_dimensions[2:]
        # Resize image to match input dimensions
        img = cv2.resize(self.img, (input_shape[1], input_shape[0]))
        # Convert image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Transpose image dimensions to match model input format and normalize pixel values
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        self.img = np.expand_dims(img, axis=0)
        #return np.expand_dims(img, axis=0)

    @staticmethod
    def compute_iou(box, boxes):
        """Intersection over Union (IoU)"""

        # Compute coordinates of intersection rectangle
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area
        return iou

    def nms(self, boxes, scores):
        """Non-maximum suppression"""
        # Sort boxes based on confidence scores
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the box with the highest confidence
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU with other boxes
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with high IoU
            keep_indices = np.where(ious < self.iou)[0]
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def boxes_coordinates(self):
        """Calculate coordinates (x1, y1, x2, y2) from points, height and width (x, y, w, h)"""
        coordinates = np.copy(self.boxes)
        coordinates[:, 0] = self.boxes[:, 0] - self.boxes[:, 2] / 2  # x1 = x - w/2
        coordinates[:, 1] = self.boxes[:, 1] - self.boxes[:, 3] / 2  # y1 = y - h/2
        coordinates[:, 2] = self.boxes[:, 0] + self.boxes[:, 2] / 2  # x2 = x + w/2
        coordinates[:, 3] = self.boxes[:, 1] + self.boxes[:, 3] / 2  # y2 = y + h/2
        self.boxes = coordinates

    def multiclass_nms(self):
        """Non-maximum suppression separately for each class to handle overlapping detections within each class"""
        unique_class_ids = np.unique(self.class_ids)
        keep_idxs = []
        for class_id in unique_class_ids:
            class_indices = np.where(self.class_ids == class_id)[0]
            class_boxes = self.boxes[class_indices, :]
            class_scores = self.scores[class_indices]

            class_keep_boxes = self.nms(class_boxes, class_scores)
            keep_idxs.extend(class_indices[class_keep_boxes])

        self.boxes = self.boxes[keep_idxs]
        self.scores = self.scores[keep_idxs]
        self.class_ids = self.class_ids[keep_idxs]

    def rescale(self):
        """Adjust boxes to original image"""
        img_h, img_w, _ = self.image_shape
        _, _, input_h, input_w = self.input_dimensions
        # Rescale boxes to original image dimensions
        self.boxes = self.boxes / np.array([input_h, input_w, input_h, input_w], dtype=np.float32)
        self.boxes *= np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        self.boxes = self.boxes.astype(int)

    def predict(self):
        input_name = self.session.get_inputs()[0].name
        results = self.session.run(None, {input_name: self.img})
        self.predictions = np.squeeze(results[0]).T

    def detect(self):
        self.preprocess()
        self.predict()

        # return empty list in case of no detected objects
        scores = np.max(self.predictions[:, 4:], axis=1)
        if scores.size == 0:
            return self.detected_objects

        # Filter out object confidence scores below threshold
        self.predictions = self.predictions[scores > self.confidence, :]
        self.scores = scores[scores > self.confidence]
        # Get class with the highest confidence for each detection
        self.class_ids = np.argmax(self.predictions[:, 4:], axis=1)
        self.boxes = self.predictions[:, :4]

        self.boxes_coordinates()
        self.multiclass_nms()
        self.rescale()

        # Create BoundingBox objects for detected objects
        for box, score, label in zip(self.boxes, self.scores, self.class_ids):
            self.detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3]))

        return self.detected_objects

