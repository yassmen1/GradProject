def expert_system_score(ans):

    score = 0

    # Critical social questions
    if ans["q1"] == 0: score += 10
    if ans["q6"] == 0: score += 10
    if ans["q7"] == 0: score += 10
    if ans["q8"] == 0: score += 10
    if ans["q10"] == 0: score += 15
    if ans["q14"] == 0: score += 15

    # Sensory
    if ans["q12"] == 1: score += 10

    # Communication
    if ans["q15"] == 0: score += 10
    if ans["q18"] == 0: score += 10

    return score