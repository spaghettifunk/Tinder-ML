__author__ = 'tobias'
import os
import re
from flask.ext.sqlalchemy import SQLAlchemy
from .config import app, photodir


db = SQLAlchemy(app)

basedir = os.path.abspath(os.path.dirname(__file__))

basephotodir = os.path.join(basedir, photodir)

def create_db(db=db):
    db.create_all()

def flush_db(db=db):
    db.drop_all()
    db.create_all()

class MeasurementData(db.Model):
    __tablename__ = 'measurementdata'
    id = db.Column(db.Integer, primary_key=True)
    session = db.Column(db.Integer)
    like = db.Column(db.Boolean)

    person_id = db.Column(db.Integer, db.ForeignKey('person.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __init__(self, session=None, like=None, person=None, user=None):
        self.session = session
        self.like = like
        self.person = person
        self.user = user

    def __repr__(self):
        return "MeasurementData: user:{}, session:{}, like:{}, photo:{}".format(self.user, self.session, self.like, self.person)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True)
    gender = db.Column(db.Integer)
    likes_gender = db.Column(db.Integer)
    measurements = db.relationship("MeasurementData", backref='user')

    def __init__(self, username=None, gender=None, likes_gender=None):
        self.username = username
        self.gender = gender
        self.likes_gender = likes_gender


    def __repr__(self):
        return "Person: name:{}, gender:{}, likes:{}".format(self.username, self.gender, self.likes_gender)


class Person(db.Model):
    __tablename__ = 'person'
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.Integer)    # 2 = Male, 1 = Female
    tinder_id = db.Column(db.String)
    photos = db.relationship("Photo",  backref='person')
    measurements = db.relationship("MeasurementData", backref='person')

    def __init__(self, tinder_id=None, gender=None):
        self.tinder_id = tinder_id
        self.gender = gender

    def __repr__(self):
        return "Person: id:{}, tinder_id:{}, gender:{}".format(self.id, self.tinder_id, self.gender)

class Photo(db.Model):
    __tablename__ = 'photo'
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String)
    person_on_picture = db.Column(db.Integer, db.ForeignKey('person.id'))


    def __init__(self, filepath=None, person=None):
        self.filepath = filepath
        self.person = person

    def __repr__(self):
        return "Photo: id:{}, filepath:{}, person: {}".format(self.id, self.filepath, self.person)

def create_photo_db(gender=0, dir=basephotodir, db=db):
    splitstring = '_'
    regex = re.compile(splitstring)
    for folder in os.listdir(dir):
        if folder != '.DS_Store':
            folderpath = os.path.join(dir, folder)
            person = Person(tinder_id=regex.split(folderpath)[1], gender=gender)
            print(person)
            db.session.add(person)
            for file in os.listdir(folderpath):
                if ('_cropped' not in file) and file != '.DS_Store':
                    path = os.path.join(folder, file)
                    photo = Photo(filepath=path, person=person)
                    db.session.add(photo)
            db.session.commit()


