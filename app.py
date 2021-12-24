from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from kobert_text_classification import BERTClassifier
import config

db = SQLAlchemy()
migrate = Migrate()



def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # ORM
    db.init_app(app)
    migrate.init_app(app, db)
    import models

    # 블루프린트
    from views import main_views
    app.register_blueprint(main_views.bp)

    return app