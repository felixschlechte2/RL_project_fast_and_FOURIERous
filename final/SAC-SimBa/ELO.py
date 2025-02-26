def Elo(rating_1,rating_2,winner):
    #winner = 1 if player 1 wins, 0.5 draw, 0 player 1 loses  


    if rating_1 < 2400:
        K_1 = 20
    else:
        K_1 = 10

    if rating_2 < 2400:
        K_2 = 20
    else:
        K_2 = 10

    
    expected_score_1 = 1/(1+10**((rating_2-rating_1)/400))
    expected_score_2 = 1/(1+10**((rating_1-rating_2)/400))

    rating_1 = rating_1 + K_1*(winner - expected_score_1)
    rating_2 = rating_2 + K_2*((1-winner) - expected_score_2)
    
    return rating_1, rating_2

def decay_elos(Elos, Decay= 0.99):
    for i in range(len(Elos)):
        Elos[i] *= Decay
    return Elos