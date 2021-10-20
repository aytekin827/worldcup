from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.orm import backref, relationship

db = SQLAlchemy()
migrate = Migrate()

class Results(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer(),primary_key=True)
    date = db.Column(db.Date(),nullable=False)
    home_team = db.Column(db.String(128),nullable=False)
    away_team = db.Column(db.String(128),nullable=False)
    home_score = db.Column(db.Integer(),nullable=False)
    away_score = db.Column(db.Integer(),nullable=False)
    tournament = db.Column(db.String(128))
    city = db.Column(db.String(128))
    country = db.Column(db.String(128))
    neutral = db.Column(db.String(128))
    
    def __repr__(self):
        return f"date : {self.date}, (home){self.home_team}{self.home_score} vs (away){self.away_team}{self.away_score}"

class ranking(db.Model):
    __tablename__ = 'ranking'

    rank = db.Column(db.Integer(),primary_key=True)
    country_full = db.Column(db.String(128),nullable=False)
    country_abrv = db.Column(db.String(128),nullable=False)
    total_points = db.Column(db.Float(),nullable=False)
    previous_points = db.Column(db.Float(),nullable=False)
    rank_change = db.Column(db.Integer(),nullable=False)
    confederation = db.Column(db.String(128),nullable=False)
    rank_date = db.Column(db.Date(),nullable=False)


class fifa_ranking(db.Model):
    __tablename__ = 'fifa_ranking'

    index = db.Column(db.Integer(),primary_key=True)
    rank = db.Column(db.Integer(),nullable=False)
    country_full = db.Column(db.String(128),nullable=False)
    country_abrv = db.Column(db.String(128),nullable=False)
    total_points = db.Column(db.Float())
    previous_points = db.Column(db.Float())
    rank_change = db.Column(db.Float())
    cur_year_avg = db.Column(db.Float())
    cur_year_avg_weighted = db.Column(db.Float())
    last_year_avg = db.Column(db.Float())
    last_year_avg_weighted = db.Column(db.Float())
    two_year_ago_avg = db.Column(db.Float())
    two_year_ago_weighted = db.Column(db.Float())
    three_year_ago_avg = db.Column(db.Float())
    three_year_ago_weighted = db.Column(db.Float())
    confederation = db.Column(db.String(128),nullable=False)
    rank_date = db.Column(db.Date(),nullable=False)

class Worldcup2018(db.Model):
    __tablename__ = 'worldcup2018'

    team = db.Column(db.String(128),primary_key=True)
    Group = db.Column(db.String(128),nullable=False)
    first_match = db.Column(db.String(128),nullable=False)
    second_match = db.Column(db.String(128),nullable=False)
    third_match = db.Column(db.String(128),nullable=False)


class models(db.Model):
    __tablename__ = 'models'

    index = db.Column(db.Integer(),primary_key=True)
    average_rank = db.Column(db.Float(),nullable=False)
    rank_difference = db.Column(db.Integer(),nullable=False)
    point_difference = db.Column(db.Float(),nullable=False)
    is_stake = db.Column(db.Boolean(),nullable=False)
    is_one = db.Column(db.Boolean(),nullable=False)

class rankings_feature(db.Model):
    __tablename__ = 'rankings_feature'

    index = db.Column(db.Integer(),primary_key=True)
    rank_date = db.Column(db.Date(),nullable=False)
    rank = db.Column(db.Float(),nullable=False)
    country_full = db.Column(db.String(128),nullable=False)
    country_abrv = db.Column(db.String(128),nullable=False)
    cur_year_avg_weighted = db.Column(db.Float(),nullable=False)
    two_year_avg_weighted = db.Column(db.Float(),nullable=False)
    three_year_avg_weighted = db.Column(db.Float(),nullable=False)
    weighted_points = db.Column(db.Float(),nullable=False)