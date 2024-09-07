# if you are using gpu for prediction, please see https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory for restricting memory usage

from nudenet import NudeClassifier
classifier = NudeClassifier()
a=classifier.classify('C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\data\\nude_data.mp4')
print(a)
# {'path_to_nude_image': {'safe': 5.8822202e-08, 'nude': 1.0}}
#
# from nudenet import NudeDetector
# detector = NudeDetector('detector_checkpoint_path')
#
# # Performing detection
# detector.detect('path_to_nude_image')
# # [{'box': [352, 688, 550, 858], 'score': 0.9603578, 'label': 'BELLY'}, {'box': [507, 896, 586, 1055], 'score': 0.94103414, 'label': 'F_GENITALIA'}, {'box': [221, 467, 552, 650], 'score': 0.8011624, 'label': 'F_BREAST'}, {'box': [359, 464, 543, 626], 'score': 0.6324697, 'label': 'F_BREAST'}]
#
# # Censoring an image
# detector.censor('path_to_nude_image', out_path='censored_image_path', visualize=False)