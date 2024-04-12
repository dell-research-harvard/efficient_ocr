import os
import base64
import requests
import json
from glob import glob

def _analyze_image(image_path, API_KEY, API_URL):
    ''' 
    Sends a request to the Google Cloud Vision API to analyze a single image
    '''
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    base64_image = base64.b64encode(content).decode('UTF-8')
    request_body = {
        "requests": [
            {
                "image": {"content": base64_image},
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }
    response = requests.post(API_URL, json=request_body, params={'key': API_KEY})
    return response.json()

def analyze_images(API_KEY: str, image_directory: str, output_directory: str, API_URL: str = 'https://vision.googleapis.com/v1/images:annotate'):
    ''' 
    Analyzes all images in a directory using the Google Cloud Vision API
    Outputs: 1 JSON file per image with the API response
    '''
    # Ensure output directory exists.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over images in the directory and send them to the API.
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f'Analyzing {filename}...')
            image_path = os.path.join(image_directory, filename)
            response_data = _analyze_image(image_path, API_KEY, API_URL)

            # Save the response in a JSON file.
            output_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}.json')
            with open(output_path, 'w') as output_file:
                json.dump(response_data, output_file)

    print('Analysis complete.')

def gcv_output_to_coco(image_directory: str, json_directory: str, output_directory: str):
    '''
    Converts Google Cloud Vision API output to COCO format 
    Outputs: 1 JSON file with the COCO dataset
    '''

    # make the directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize COCO dataset structure
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "word"},
            {"id": 0, "name": "character"}
        ]
    }

    # Helper function to convert bounding box format
    def _convert_bbox(vertices):

        x = vertices[0]['x']
        y = vertices[0]['y']
        w = vertices[1]['x'] - x
        h = vertices[2]['y'] - y

        return [x, y, w, h]
    
    # Processing each JSON file

    annotation_id = 1 
    for json_file in glob(os.path.join(json_directory, '*.json')):
        with open(json_file) as file:
            data = json.load(file)
        
            # Assuming each JSON file corresponds to one image
            image_name = os.path.basename(json_file).replace('.json', '')
            image_path = os.path.join(image_directory, image_name)
            image_id = image_name

            # Add image information with 'text' field
            image_info = data["responses"][0]["fullTextAnnotation"]["pages"][0]
            full_text = data["responses"][0]["fullTextAnnotation"]["text"]

            # Replace illegal chars
            full_text = full_text.replace('\n', ' ')
            full_text = full_text.replace('-', '')
            full_text = full_text.replace('–', '')
            full_text = full_text.replace('—', '')

            coco_dataset["images"].append({
                "id": f"{image_id}.png",
                "file_name": f"{image_name}.png",
                "width": image_info["width"],
                "height": image_info["height"],
                "text": full_text
            })

            # Add annotations for words with text label
            for annotation in data["responses"][0]["textAnnotations"][1:]:
                bbox = _convert_bbox(annotation["boundingPoly"]["vertices"])
                text = annotation["description"]

                # Replace illegal chars (this is redundant I think)
                text = text.replace('\n', ' ')
                text = text.replace('-', '')
                text = text.replace('–', '')
                text = text.replace('—', '')

                coco_dataset["annotations"].append({
                    "id": annotation_id,
                    "image_id": f"{image_id}.png",
                    "category_id": 1,
                    "bbox": bbox,
                    "segmentation": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3]],
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "text": text
                })
                annotation_id += 1

            # Add annotations for individual letters with text label
            for block in image_info["blocks"]:
                for paragraph in block["paragraphs"]:
                    for word in paragraph["words"]:
                        for symbol in word["symbols"]:
                            bbox = _convert_bbox(symbol["boundingBox"]["vertices"])
                            text = symbol["text"]

                            # If the annotation is an illegal character, don't add it
                            if text == ' ' or text == '\n' or text == '-' or text == '–' or text == '—':
                                continue

                            coco_dataset["annotations"].append({
                                "id": annotation_id,
                                "image_id": f"{image_id}.png",
                                "category_id": 0,
                                "bbox": bbox,
                                "segmentation": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3]],
                                "area": bbox[2] * bbox[3],
                                "iscrowd": 0,
                                "text": text
                            })
                            annotation_id += 1

    # Write to output directory
    # Determine output file name and path
    output_file_name = "COCO_dataset.json"
    output_file_path = os.path.join(output_directory, output_file_name)

    # Write COCO dataset to JSON file in output directory
    with open(output_file_path, 'w') as output_file:
        json.dump(coco_dataset, output_file)
        

    print(f"COCO json created at {output_file_path}")