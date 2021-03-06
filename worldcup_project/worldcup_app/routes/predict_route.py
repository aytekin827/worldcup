from itertools import combinations
from flask import Blueprint, render_template, request
from worldcup_app.models import models,Worldcup2018,rankings_feature
from worldcup_app.views.model_views import logistic_regression, single_elimination_round,split,simulation_groupstage
import pandas as pd

bp = Blueprint('predict', __name__)

@bp.route('/predict')
def index():
    X = models.query.with_entities(models.average_rank, models.rank_difference, models.point_difference, models.is_stake).all()
    y = models.query.with_entities(models.is_one).all()
    X_train, X_test, y_train, y_test = split(X,y)
    model = logistic_regression(X_train,y_train)

    data_worldcup = Worldcup2018.query.with_entities(
        Worldcup2018.team, 
        Worldcup2018.Group, 
        Worldcup2018.first_match, 
        Worldcup2018.second_match, 
        Worldcup2018.third_match).all()
    world_cup = pd.DataFrame(data_worldcup,columns=[
        'team',
        'Group',
        'first_match',
        'second_match',
        'third_match'])

    data_ranking = rankings_feature.query.with_entities(
        rankings_feature.rank_date,
        rankings_feature.rank,
        rankings_feature.country_full,
        rankings_feature.country_abrv,
        rankings_feature.cur_year_avg_weighted,
        rankings_feature.two_year_avg_weighted,
        rankings_feature.three_year_avg_weighted,
        rankings_feature.weighted_points).all()
    rankings = pd.DataFrame(data_ranking,columns=[
        'rank_date',
        'rank',
        'country_full',
        'country_abrv',
        'cur_year_avg_weighted',
        'two_year_avg_weighted',
        'three_year_avg_weighted',
        'weighted_points'
    ])


    result_table,keys,world_cup,world_cup_rankings = simulation_groupstage(world_cup,rankings,model)
    single_result_table,odd,lables,finals = single_elimination_round(world_cup,world_cup_rankings,model)
    top2 = lambda x: x.sort_values(by='points', ascending=False)[:4] 
    df = world_cup.groupby(by='Group').apply(top2)
    return render_template(
        'predict.html',
        result = result_table,
        keys = keys,
        df=df,
        single_result_table=single_result_table,
        finals = finals
        )

# @bp.route('/predict/model')
# def index():
#     dataset = models.query.all()
#     return render_template('predict_html',dataset=dataset)