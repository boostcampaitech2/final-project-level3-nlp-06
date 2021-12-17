from app import db

class Stocks(db.Model):

    id = db.Column(db.Integer, primary_key=True, unique=True)
    news = db.Column(db.Text)
    keywords = db.Column(db.TEXT)
    corporations = db.Column(db.TEXT)
    memo = db.Column(db.TEXT)

class Evaluations(db.Model):

    id = db.Column(db.Integer, primary_key=True, unique=True)
    count = db.Column(db.Integer)
    score = db.Column(db.TEXT)
    mean = db.Column(db.Float)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id', ondelete='CASCADE'))
    stock = db.relationship('Stocks', backref=db.backref('eval_set', cascade='all, delete-orphan'))

class News(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True)
    corporation = db.Column(db.String(50))
    news = db.Column(db.TEXT)
    corporation_id = db.Column(db.Integer, db.ForeignKey('corporations.id', ondelete='CASCADE'))
    corporation = db.relationship('Corporations', backref=db.backref('news_set', cascade='all, delete-orphan'))

class Corporations(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(50))
    details = db.Column(db.TEXT)
    summaries = db.Column(db.TEXT)
