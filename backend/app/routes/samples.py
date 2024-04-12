from flask import Blueprint, request, current_app
from app.services.samples_service import Samples
from app.utils.utils import create_sample_data_directory

samples_blueprint = Blueprint('samples', __name__)

@samples_blueprint.route('/tokens')
def get_tokens():
    samples_class = current_app.sample_service
    sample = request.args.get('sample', False)
    sample_idx = int(request.args.get('sample_idx'))
    if sample == "":
        return

    if sample_idx >= len(samples_class.samples):
        samples_class.add(sample)
    
    else:
        samples_class.replace(sample, sample_idx)

    create_sample_data_directory(sample_idx)
    tokens = samples_class.untokenize_sample_tokenized(sample_idx)
    return {"tokens": tokens, "n_tokens": len(tokens)}

@samples_blueprint.route('/forward')
def forward():
    samples_class = current_app.sample_service
    sample_idx = int(request.args.get('sample_idx'))
    samples_class.forward(sample_idx)
    print(sample_idx)

    return {"Status": "Done"}

@samples_blueprint.route('/activations/histogram/<string:type>/<int:layer>/<int:token>')
def get_histogram_activations(type, layer, token):
    samples_class = current_app.sample_service
    attn_head = request.args.get('attn_head', None)
    rope_mode = request.args.get('rope_mode', "full")
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_histogram_activations_layer_token(sample_idx, type, layer, token, attn_head, rope_mode)

    return return_object

@samples_blueprint.route('/highlight/<string:type>/<int:layer>/<int:token>/<int:dim>')
def get_activation_value_histogram(type, layer, token, dim):
    samples_class = current_app.sample_service
    attn_head = request.args.get('attn_head', None)
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_activation_value_histogram(sample_idx, dim, type, layer, token, attn_head)

    return return_object

@samples_blueprint.route('/predictions')
def get_top_predictions():
    samples_class = current_app.sample_service
    top = int(request.args.get('top', 5))
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_top_predictions(sample_idx, top)

    return return_object

@samples_blueprint.route('/attention/<int:layer>/<int:attn_head>/<int:token>')
def get_attention(layer, attn_head, token):
    samples_class = current_app.sample_service
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_attention(sample_idx, layer, attn_head, token)

    return return_object

@samples_blueprint.route('/attention_opposite/<int:layer>/<int:attn_head>/<int:token>')
def get_attention_opposite(layer, attn_head, token):
    samples_class = current_app.sample_service
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_attention_opposite(sample_idx, layer, attn_head, token)

    return return_object

@samples_blueprint.route('/loss')
def get_losses():
    samples_class = current_app.sample_service
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_losses(sample_idx)

    return return_object

@samples_blueprint.route('/similarities_tokens/<string:type>/<int:layer>/<int:token>')
def get_similarities_tokens(type, layer, token):
    samples_class = current_app.sample_service
    attn_head = request.args.get('attn_head', None)
    rope_mode = request.args.get('rope_mode', "full")
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_similarities_tokens(sample_idx, type, layer, attn_head, token, rope_mode)

    return return_object

@samples_blueprint.route('/similarities_previous/<string:type>/<int:layer>')
def get_similarities_previous(type, layer):
    samples_class = current_app.sample_service
    attn_head = request.args.get('attn_head', None)
    rope_mode = request.args.get('rope_mode', "full")
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_similarities_previous(sample_idx, type, layer, attn_head, rope_mode)

    return return_object

@samples_blueprint.route('/similarities_previous_residual/<string:type>/<int:layer>')
def get_similarities_previous_residual(type, layer):
    samples_class = current_app.sample_service
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_similarities_previous_residual(sample_idx, type, layer)

    return return_object

@samples_blueprint.route('/activations_values/<string:type>/<int:layer>')
def get_activations_values(type, layer):
    samples_class = current_app.sample_service
    attn_head = request.args.get('attn_head', None)
    sample_idx = int(request.args.get('sample_idx'))
    activation_idx = int(request.args.get('activation_idx'))
    return_object = samples_class.get_activations_values(sample_idx, type, layer, attn_head, activation_idx)

    return return_object

@samples_blueprint.route('/activations_sum/<string:type>/<int:layer>')
def get_activations_sum(type, layer):
    samples_class = current_app.sample_service
    attn_head = request.args.get('attn_head', None)
    sample_idx = int(request.args.get('sample_idx'))
    return_object = samples_class.get_activations_sum(sample_idx, type, layer, attn_head)

    return return_object

@samples_blueprint.route('/activation_vector/<string:type>/<int:layer>')
def get_activation_vector(type, layer):
    samples_class = current_app.sample_service
    attn_head = request.args.get('attn_head', None)
    sample_idx = int(request.args.get('sample_idx'))
    token_idx = int(request.args.get('token_idx'))
    return_object = samples_class.get_activation_vector(sample_idx, type, layer, attn_head, token_idx)

    return return_object