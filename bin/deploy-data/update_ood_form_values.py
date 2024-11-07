import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.settings import (supported_languages, 
                          available_whisper_models, 
                          default_whisper_model)

# Path to the form.yml file
form_yml_path = 'bin/deploy-data/ood/form1.yml'

# Load the existing form.yml
with open(form_yml_path, 'r') as file:
    form_data = yaml.safe_load(file)

# Update the language_field options with supported languages
form_data['attributes']['language_field']['options'] = list(supported_languages.keys())

# Update the model_selector options with available whisper models
form_data['attributes']['model_selector']['options'] = [
    ["default", default_whisper_model]
] + [[model, model] for model in available_whisper_models]

# Write the updated data back to form.yml
with open(form_yml_path, 'w') as file:
    yaml.dump(form_data, file, default_flow_style=False, sort_keys=False)

print("form.yml has been updated with supported languages and Whisper models from settings.py.")