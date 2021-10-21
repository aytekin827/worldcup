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

    result_table = dict()
    for group in set(world_cup['Group']):
        result_table[group] = []
        for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
            print(group,home,away)
            row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=['average_rank', 'rank_difference', 'point_difference', 'is_stake'])
            home_rank = world_cup_rankings.loc[home, 'rank']
            home_points = world_cup_rankings.loc[home, 'weighted_points']
            opp_rank = world_cup_rankings.loc[away, 'rank']
            opp_points = world_cup_rankings.loc[away, 'weighted_points']
            row['average_rank'] = (home_rank + opp_rank) / 2
            row['rank_difference'] = home_rank - opp_rank
            row['point_difference'] = home_points - opp_points
            
            home_win_prob = model.predict_proba(row)[:,1][0]
            world_cup.loc[home, 'total_prob'] += home_win_prob
            world_cup.loc[away, 'total_prob'] += 1-home_win_prob
            
            # awayteam win
            if home_win_prob <= 0.5 - margin:
                world_cup.loc[away, 'points'] += 3
                res = f'{home} vs {away}: {away} wins with {1-home_win_prob:2f}'
                
            # draw
            elif 0.5 + margin > home_win_prob > 0.5 - margin:
                world_cup.loc[home, 'points'] += 1
                world_cup.loc[away, 'points'] += 1
                res = f'{home} vs {away}: Draw'
                            
            # hometeam win
            else:
                world_cup.loc[home, 'points'] += 3
                res = f'{home} vs {away}: {home} wins with {home_win_prob:2f}'
                
            result_table[group].append(res)
    sort_dict_keys = sorted(result_table.keys())

    return result_table, sort_dict_keys, world_cup, world_cup_rankings

def single_elimination_round(world_cup,world_cup_rankings,model):
    pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]

    world_cup = world_cup.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
    next_round_wc = world_cup.groupby('Group').nth([0, 1]) # select the top 2
    next_round_wc = next_round_wc.reset_index()
    next_round_wc = next_round_wc.loc[pairing]
    next_round_wc = next_round_wc.set_index('team')

    finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

    labels = list()
    odds = list()
    result_table=dict()

    for f in finals:
        iterations = int(len(next_round_wc) / 2)
        winners = []
        result_table[f] = []
        for i in range(iterations):
            home = next_round_wc.index[i*2]
            away = next_round_wc.index[i*2+1]

            row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=['average_rank', 'rank_difference', 'point_difference', 'is_stake'])
            home_rank = world_cup_rankings.loc[home, 'rank']
            home_points = world_cup_rankings.loc[home, 'weighted_points']
            opp_rank = world_cup_rankings.loc[away, 'rank']
            opp_points = world_cup_rankings.loc[away, 'weighted_points']
            row['average_rank'] = (home_rank + opp_rank) / 2
            row['rank_difference'] = home_rank - opp_rank
            row['point_difference'] = home_points - opp_points

            home_win_prob = model.predict_proba(row)[:,1][0]
            if model.predict_proba(row)[:,1] <= 0.5:
                winners.append(away)
                result_table[f].append("{0} vs. {1}: {2} wins with probability {3:.2f}".format(home,away,away, 1-home_win_prob))
            else:
                winners.append(home)
                result_table[f].append("{0} vs. {1}: {2} wins with probability {3:.2f}".format(home,away,home, home_win_prob))

            labels.append("{}({:.2f}) vs. {}({:.2f})".format(world_cup_rankings.loc[home, 'country_abrv'], 
                                                            home_win_prob, 
                                                            world_cup_rankings.loc[away, 'country_abrv'], 
                                                            (1-home_win_prob)))
            odds.append([home_win_prob, 1-home_win_prob])
                    
        next_round_wc = next_round_wc.loc[winners]

    return result_table, odds, labels, finals

def result_visualization(odds,labels):
    import networkx as nx
    import pydot
    from networkx.drawing.nx_pydot import graphviz_layout
    import matplotlib as plt

    node_sizes = pd.DataFrame(list(reversed(odds)))
    scale_factor = 0.3 # for visualization
    G = nx.balanced_tree(2, 3)
    pos = graphviz_layout(G, prog='twopi')
    centre = pd.DataFrame(pos).mean(axis=1).mean()

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(1,1,1)
    # add circles 
    circle_positions = [(235, 'black'), (180, 'blue'), (120, 'red'), (60, 'yellow')]
    [ax.add_artist(plt.Circle((centre, centre), 
                            cp, color='grey', 
                            alpha=0.2)) for cp, c in circle_positions]

    # draw first the graph
    nx.draw(G, pos, 
            node_color=node_sizes.diff(axis=1)[1].abs().pow(scale_factor), 
            node_size=node_sizes.diff(axis=1)[1].abs().pow(scale_factor)*2000, 
            alpha=1, 
            cmap='Reds',
            edge_color='black',
            width=10,
            with_labels=False)

    # draw the custom node labels
    shifted_pos = {k:[(v[0]-centre)*0.9+centre,(v[1]-centre)*0.9+centre] for k,v in pos.items()}
    nx.draw_networkx_labels(G, 
                            pos=shifted_pos, 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5, alpha=1),
                            labels=dict(zip(reversed(range(len(labels))), labels)))

    texts = ((10, 'Best 16', 'black'), (70, 'Quarter-\nfinal', 'blue'), (130, 'Semifinal', 'red'), (190, 'Final', 'yellow'))
    [plt.text(p, centre+20, t, 
            fontsize=12, color='grey', 
            va='center', ha='center') for p,t,c in texts]
    plt.axis('equal')
    plt.title('Single-elimination phase\npredictions with fair odds', fontsize=20)
    plt.show()