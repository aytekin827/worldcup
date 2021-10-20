from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import pandas as pd
import numpy as np

def logistic_regression(X_train,y_train):
    logreg = linear_model.LogisticRegression(C=1e-5)
    features = PolynomialFeatures(degree=2)
    model = Pipeline([
        ('polynomial_features', features),
        ('logistic_regression', logreg)
    ])
    model = model.fit(X_train, y_train)
    return model

def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def simulation_groupstage(world_cup,rankings,model,margin = 0.05):
    world_cup = world_cup.dropna(how='all')
    world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "Korea Republic"})
    world_cup = world_cup.set_index('team')

    world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                    rankings['country_full'].isin(world_cup.index.unique())]
    world_cup_rankings = world_cup_rankings.set_index(['country_full'])

    opponents = ['First match', 'econd match', 'third match']

    world_cup['points'] = 0
    world_cup['total_prob'] = 0

    # for group in set(world_cup['Group']):
    #     print('___Starting group {}:___'.format(group))
    #     for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
    #         print("{} vs. {}: ".format(home, away), end='')
    #         row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=['average_rank', 'rank_difference', 'point_difference', 'is_stake'])
    #         home_rank = world_cup_rankings.loc[home, 'rank']
    #         home_points = world_cup_rankings.loc[home, 'weighted_points']
    #         opp_rank = world_cup_rankings.loc[away, 'rank']
    #         opp_points = world_cup_rankings.loc[away, 'weighted_points']
    #         row['average_rank'] = (home_rank + opp_rank) / 2
    #         row['rank_difference'] = home_rank - opp_rank
    #         row['point_difference'] = home_points - opp_points
            
    #         home_win_prob = model.predict_proba(row)[:,1][0]
    #         world_cup.loc[home, 'total_prob'] += home_win_prob
    #         world_cup.loc[away, 'total_prob'] += 1-home_win_prob
            
    #         points = 0
    #         if home_win_prob <= 0.5 - margin:
    #             print("{} wins with {:.2f}".format(away, 1-home_win_prob))
    #             world_cup.loc[away, 'points'] += 3
    #         if home_win_prob > 0.5 - margin:
    #             points = 1
    #         if home_win_prob >= 0.5 + margin:
    #             points = 3
    #             world_cup.loc[home, 'points'] += 3
    #             print("{} wins with {:.2f}".format(home, home_win_prob))
    #         if points == 1:
    #             print("Draw")
    #             world_cup.loc[home, 'points'] += 1
    #             world_cup.loc[away, 'points'] += 1
    #     print()
    return world_cup