from app.routes.samples import samples_blueprint
from app.routes.general import general_blueprint

def configure_routes(app):
    app.register_blueprint(general_blueprint, url_prefix='/api/general')
    app.register_blueprint(samples_blueprint, url_prefix='/api/samples')