from flask import Flask
import os

# CSV파일 경로
CSV_FILEPATH = os.path.join(os.getcwd(),__name__,'results.csv')

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:1234@localhost:5433/worldcup"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['DEBUG'] = True

    from models import db, migrate

    db.init_app(app)
    migrate.init_app(app,db)

    from routes import main_route, info_route, predict_route
    app.register_blueprint(main_route.bp)
    app.register_blueprint(info_route.bp)
    app.register_blueprint(predict_route.bp,url_prefix='/api')

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

