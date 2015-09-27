__author__ = 'tobias'
import os
from flask import Flask

photodir = 'photos'


def config_app():
    app = Flask(__name__)
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] =\
    'sqlite:///' + os.path.join(basedir, 'data.sqlite')
    app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
    app.config['basedir'] = basedir
    app.secret_key = 'thanoe3oavejon1gaveiRaihohj5Yaht3aishoChuax4hutied'
    app.config['APP_PASSWORD'] = 'quaikiK5Oh'

    return app

app = config_app()