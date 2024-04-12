from flask import Flask
from app.models.model import Model
from app.services.samples_service import Samples
from app.routes.general import general_blueprint
from app.routes.samples import samples_blueprint

def create_app():
    app = Flask(__name__)
    print('Loading model...')
    app.model_class = Model()
    print('Model loaded')
    print(f"{len(app.model_class.model_layers)} layers")

    app.sample_service = Samples(model_class=app.model_class)

    from app.routes import configure_routes
    configure_routes(app)

    return app
