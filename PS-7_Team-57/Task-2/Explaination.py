import os
from PIL import Image
import json
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch
from transformers import BitsAndBytesConfig
import requests

# Function to load the JSON file containing artifact explanations
def load_json_file(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)

# Function to retrieve image for the given index
def get_image(index, images_directory):
    filename = f"{index}.png"  # Assuming image files are named by their index
    image_path = os.path.join(images_directory, filename)

    if os.path.exists(image_path):
        image = Image.open(image_path)  # Open the image using PIL
        return image
    else:
        print(f"Image for index {index} not found: {image_path}")
        return None

# Function to retrieve images and corresponding artifacts for all indexes in JSON
def get_images_and_artifacts(json_file_path, images_directory):
    # Load the JSON data
    data = load_json_file(json_file_path)

    results = []

    # Loop through all entries in the JSON file
    for entry in data:
        index = entry['index']

        # Retrieve the image corresponding to the index
        image = get_image(index, images_directory)

        if image:
            result = {
                "index": index,
                "image": image,  # The image object (PIL Image)
                "artifacts": entry['explanation']  # List of artifact explanations
            }
            results.append(result)
        else:
            print(f"Skipping index {index} as the image is not found.")

    return results

# Example usage
json_file_path = "/content/file1.json"  # Path to your input JSON file
images_directory = "/content/test-interiit/perturbed_images_32"  # Path to the directory where images are stored

results = get_images_and_artifacts(json_file_path, images_directory)

# Example to process results
for result in results:
    print(f"Artifacts for index {result['index']}: {result['artifacts']}")
    result['image'].show()  # Show the image using PIL





device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)


processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
model = Idefics2ForConditionalGeneration.from_pretrained(
      "HuggingFaceM4/idefics2-8b",
      torch_dtype=torch.float16,
      quantization_config=bnb_config
  )
model.to(device)


# Initialize a list to store all the results
all_results = []

# Assuming results contains the data for processing
for result in results:
    images = [result['image']]


    sys_msg = f'The image contains the following artifacts: {result["artifacts"]}. ' \
    'Please explain each of these artifacts in one sentence, providing a reference to how they appear in the image. ' \
    'For each artifact, describe its characteristics, its impact on the visual quality of the image, and any specific details in the image that showcase the artifact. ' \
    'Make sure to explain how each artifact is represented visually in the image, and discuss why these artifacts are important in terms of the image\'s overall composition, realism, or other relevant aspects.'\
    'the generated text should be in the format: {Artifact_1: Explanation_1:}  and so on in different lines, each explanation in one sentence, do not mention the name of artifact it is explained for.'\
    'make sure to explain each and every artifact in the correct specified format. '
    'For example: Cinematization Effects: The image has a cinematic effect, where the colors are vivid and the focus is on the people and the background. Movie-poster like composition of ordinary scenes: The image has a movie-poster like composition, where the people and the background are arranged in a way that looks like a movie poster. Exaggerated characteristic features: The people in the image have exaggerated characteristic features, such as their facial features and their clothing. Ghosting effects: Semi-transparent duplicates of elements: The image has ghosting effects, where semi-transparent duplicates of elements are visible, such as the people and the background. Unrealistic specular highlights: The image has unrealistic specular highlights, where the highlights are not realistic and look unnatural.Aliasing along high-contrast edges: The image has aliasing along high-contrast edges, where the edges of the people and the background are not smooth and have a jagged appearance.'
    'Give output in this exact format specified in the example.'
    'strictly give the explanation in one line and don\'t use numbering.'

    messages = [
      {
        "role":"system",
         "content":[
             {"type":"text","text":sys_msg}
        ]
      },
      {

        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze the image~"},
            {"type": "image"},
          ],
      },
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)



    inputs = processor(images=images, text=text, return_tensors="pt").to(device)


    generated_text = model.generate(**inputs, max_new_tokens=500)
    generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
    generated_text = generated_text.split('Assistant:')[-1]

    print(generated_text)
    # Prepare the artifact explanations
    generated_text = generated_text.split('.')

    dict_ = {}
    for gen in generated_text:
      gen = gen.split(':')
      for i in range(len(gen)):
        dict_[gen[0].strip()] = gen[-1]

    print(dict_)
    # Add the result to the list
    all_results.append({
        "index": result['index'],
        "explanation": dict_
    })
    #print(all_results)
# Save all results into a single JSON file
with open('artifact_explanations.json', 'w') as json_file:
    json.dump(all_results, json_file, indent=4)

print("Results saved to artifact_explanations.json")