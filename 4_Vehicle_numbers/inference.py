import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from cv2 import rectangle

from utils import *
from settings import MODELS_PATH, THRESHOLD, BRAND_CLASSES


class LetterRecognition:
    def __init__(self, model_name: str = "letter.pb"):
        self.detector = load_detection_graph(MODELS_PATH, model_name=model_name)

    def predict(self, image: np.array, threshold: str = THRESHOLD):
        """License plate recognition predict implementation.

        inference for car, plate, letter detection models
        Args:
            image: <np.array>
            threshold: threshold for detection accuracy

        Returns:
             List[license_number, [point1, point2], [point1, point2], ... ] for letter task
        """
        image_expanded = np.expand_dims(image, axis=0)
        results = []
        detector = self.detector
        # get results from network by given image
        boxes, scores, classes, num_detections, _ = detector["tf_session"].run(
            detector["tensor_names"],
            feed_dict={detector["image_tensor"]: image_expanded},
        )

        # check number of returned detections
        if num_detections > 0:
            best_rects = []
            best_scores = []
            best_letters = []

            # check each detection
            for i in range(len(boxes[0])):
                box = boxes[0][i]
                score = scores[0][i]
                cls = int(classes[0][i])
                curr_rect = (
                    int(box[1] * image.shape[1]),
                    int(box[0] * image.shape[0]),
                    int(box[3] * image.shape[1]) - int(box[1] * image.shape[1]),
                    int(box[2] * image.shape[0]) - int(box[0] * image.shape[0]),
                )

                # assert by threshold and area of bounding boxes
                if score > threshold and curr_rect[2] > 0 and curr_rect[3] > 0:

                    best_index = None
                    for j in range(len(best_letters)):
                        # check intersection area between provided and best bb
                        intersection = rect_intersection(curr_rect, best_rects[j])
                        if intersection is not None:
                            ratio = rect_area(intersection) / min(
                                rect_area(best_rects[j]), rect_area(curr_rect)
                            )
                            if (ratio > 0.6) and (score > best_scores[j]):
                                best_index = j
                            elif (ratio > 0.6) and (score < best_scores[j]):
                                best_index = i

                    # if met first time
                    if best_index is None:
                        best_letters.append(LETTER_CLASSES[cls])
                        best_rects.append(curr_rect)
                        best_scores.append(score)
                    else:
                        best_letters[j] = LETTER_CLASSES[cls]
                        best_rects[j] = curr_rect
                        best_scores[j] = score

            # sort collect best letters to single string
            indices = list(range(len(best_letters)))
            indices.sort(key=lambda q: CmpRect(best_rects[q]))

            letters = []
            for i in range(len(best_letters)):
                letters_rect = [
                    best_rects[i][0],
                    best_rects[i][1],
                    best_rects[i][0] + best_rects[i][2],
                    best_rects[i][1] + best_rects[i][3],
                ]
                letters.append(letters_rect)

            result = {
                "license_number": "".join([best_letters[i] for i in indices]),
                "license_number_score": round(float(np.mean(best_scores)), 3),
                "letters": letters,
            }
            results.append(result)
        return results


class PlateRecognition:
    def __init__(self, model_name: str = "plate.pb"):
        self.detector = load_detection_graph(MODELS_PATH, model_name=model_name)

    def predict(
        self, image: np.array, threshold: str = THRESHOLD
    ) -> List[Dict[str, Union[int, List[int]]]]:
        """Recognize number plates from image

        Args:
            image: <np.array> of rgb image
            threshold: <float> range 0..1 of how strict detector should be

        Returns:
            List of detected plates dictionaries with its properties: class, b_box, score, coords
        """
        results = []
        image_expanded = np.expand_dims(image, axis=0)

        detector = self.detector
        # get results from network by given image
        boxes, scores, classes, num_detections, _ = detector["tf_session"].run(
            detector["tensor_names"],
            feed_dict={detector["image_tensor"]: image_expanded},
        )

        # check number of returned detections
        if num_detections > 0:
            # check each detection
            for i in range(len(boxes[0])):
                box = boxes[0][i]
                score = scores[0][i]
                rectangle = (
                    int(box[0] * image.shape[0]),
                    int(box[1] * image.shape[1]),
                    int(box[2] * image.shape[0]) - int(box[0] * image.shape[0]),
                    int(box[3] * image.shape[1]) - int(box[1] * image.shape[1]),
                )

                # assert by threshold and area of bounding boxes
                if score > threshold and rectangle[2] > 0 and rectangle[3] > 0:
                    xmin = rectangle[1]
                    xmax = rectangle[1] + rectangle[3]
                    ymin = rectangle[0]
                    ymax = rectangle[0] + rectangle[2]
                    coordinates = [xmin, ymin, xmax, ymax]

                    result = {
                        "plate_score": round(float(score), 3),
                        "plate_coords": coordinates,
                    }
                    results.append(result)
        return results


def nms(res, thresh):
    dets = np.array(res)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(res[i])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def process_plate_images(
    plate_detector: PlateRecognition,
    letter_detector: LetterRecognition,
    image: np.array,
) -> List[Dict[str, Union[Union[int, List[int]], Any]]]:
    """Process vehicle analysis pipeline.
    plate detector -> letter recognition

    Args:
        plate_detector: <PlateRecognition> model instance
        letter_detector: <LetterRecognition> model instance
        image: <np.array> image

    Returns:
        <List[Dict[str, Union[Union[int, List[int]], Any]]]> of results
    """
    results = []
    image_copy = image.copy()
    plate_result = plate_detector.predict(image, threshold=0.12)
    height, width, _ = image.shape
    gl_plates = []
    if plate_result:
        for plate in plate_result:

            plate_results = plate.get("plate_coords")
            plate_results.append(plate.get("plate_score", 0))
            gl_plates.append(plate_results)

        nms_plates = nms(gl_plates, 0.1)
        for nms_plate in nms_plates:
            image_result = {}
            image_result.update({"plate_coords": nms_plate[:3], "plate_score": nms_plate[4]})
            plate_image = image[
              nms_plate[1]: nms_plate[3], nms_plate[0]: nms_plate[2]
            ].copy()
            rectangle(
                image_copy,
                (nms_plate[0], nms_plate[1]),
                (nms_plate[2], nms_plate[3]),
                (0, 255, 0),
                3,
            )
            if plate_image.shape[0] == 0 or plate_image.shape[1] == 0:
                continue
            letter_result = letter_detector.predict(plate_image)
            if letter_result:
                image_result.update(letter_result[0])
                if image_result.get("letters"):
                    letters = image_result["letters"]
                    for l, letter in enumerate(letters):
                        for i, coord in enumerate(letter):
                            if (i % 2) == 0:
                                image_result["letters"][l][i] += nms_plate[1]
                            else:
                                image_result["letters"][l][i] += nms_plate[0]
                    font_scale = 3
                    font = cv2.FONT_HERSHEY_PLAIN
                    cv2.putText(
                        image_copy,
                        letter_result[0].get("license_number", ""),
                        (
                            (nms_plate[0] + 40),
                            max(50, (nms_plate[1]) - 20),
                        ),
                        font,
                        fontScale=font_scale,
                        color=(255, 0, 0),
                        thickness=3,
                    )
            results.append(image_result)
        image_copy = cv2.putText(image_copy, f"Number plates: {len(nms_plates)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return results, image_copy


if __name__ == "__main__":
    pass
