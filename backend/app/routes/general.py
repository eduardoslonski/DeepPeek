from flask import Blueprint, request, current_app
import json

general_blueprint = Blueprint('general', __name__)

@general_blueprint.route("/model-config")
def get_model_config():
    model_class = current_app.model_class
    return model_class.model_config()

@general_blueprint.route('/samples-to-select')
def get_samples_to_select():
    with open('app/data/samples_list.json', 'r') as file:
        data = json.load(file)

    ordered_data = [{"category": category, "items": items} for category, items in data.items()]

    return ordered_data

@general_blueprint.route('/sample-text/<string:dataset>/<int:id>')
def get_sample_text(dataset, id):
    with open(f'app/data/samples_text/{dataset}.json', 'r') as file:
        data = json.load(file)

    return {"text": data[id]}