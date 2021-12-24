# from database import db
from app import db

class Validations(db.Model):

    id = db.Column(db.Integer, primary_key=True, unique=True)
    news = db.Column(db.TEXT)
    keywords = db.Column(db.TEXT)
    corporations = db.Column(db.TEXT)
    memo = db.Column(db.TEXT)

class Evaluations(db.Model):

    id = db.Column(db.Integer, primary_key=True, unique=True)
    count = db.Column(db.Integer)
    score = db.Column(db.TEXT)
    mean = db.Column(db.Float)
    validation_id = db.Column(db.Integer, db.ForeignKey('validations.id', ondelete='CASCADE'))
    validations = db.relationship('Validations', backref=db.backref('eval_set', cascade='all, delete-orphan'))

class News(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True)
    news = db.Column(db.TEXT)
    corporation_id = db.Column(db.Integer, db.ForeignKey('corporations.id', ondelete='CASCADE'))
    corporations = db.relationship('Corporations', backref=db.backref('news_set', cascade='all, delete-orphan'))

class Corporations(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(50))
    details = db.Column(db.TEXT)
    summaries = db.Column(db.TEXT)
