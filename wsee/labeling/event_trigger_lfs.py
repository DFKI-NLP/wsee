from snorkel.labeling import labeling_function
from fuzzywuzzy import process
from wsee.preprocessors.preprocessors import *

Accident = 0
CanceledRoute = 1
CanceledStop = 2
Delay = 3
Obstruction = 4
RailReplacementService = 5
TrafficJam = 6
O = 7
ABSTAIN = -1

# Keyword-based LFs
public_transport_keywords = [
    'Oberleitungsschaden', 'Signalstörung', 'Weichenstörung', 'Stellwerksstörung', 'Bahnstreik',
    'Fahrzeugstörung', 'Bahnübergangsstörung', 'Stellwerksausfall', 'Teilausfall', 'Teilausfällen', 'Ampelstörung',
    'Triebwagenstörung'
]
intervention_keywords = [
    'Intervention', 'Notarzteinsatz', 'Notarzt', 'Notarzteinsatzes', 'Polizeieinsatz',
    'Polizeieinsatzes', 'Feuerwehreinsatz', 'Feuerwehreinsatzes'
]
accident_keywords = [
    'Unfall', 'Unfälle', 'Unfalles', 'verunglückt', 'Zusammenstoß', 'Massenkarambolage', 'Kollision', 'Unglück',
    'Tote', 'Verletzte', 'verletzt', 'Toter', 'Tödliche Verletzungen', 'kollidiert', 'kollidieren', 'Schaden'
]
accident_exact_keywords = [
    'Zugkollision', 'Zugunglück'
]
accident_lower_priority_keywords = [
    'Geisterfahrer', 'Falschfahrer', 'Personenschaden', 'Bergungsarbeit', 'Bergungsarbeiten'
]
canceledroute_keywords = [
    'fällt aus', 'ausgefallen', 'fällt', 'fallen', 'unterbrochen', 'Ausfälle', 'Einstellung', 'Ausfall', 'streichen',
    'gestrichen', 'Streckensperrung', 'gesperrt', 'entfallen', 'entfällt', 'faellt', 'kein Zugverkehr'
]
canceledroute_exact_keywords = [
    'Zugausfall', 'Zugausfälle', 'Zugausfällen', 'S-Bahn-Ausfall', 'Teilausfall', 'Teilausfällen', 'Fahrtausfälle',
    'Fahrtausfällen', 'keine S-Bahnen', 'kein S-Bahnverkehr', 'Ausfall von S-Bahnen', 'kein #Zugverkehr'
]
canceledstop_keywords = [
    'hält nicht', 'halten nicht', 'hält', 'geschlossen', 'gesperrt', 'kein Halt'
]
canceledstop_exact_keywords = [
    'entfallen', 'entfällt', 'Halt entfällt', 'entfaellt', 'Haltausfall', 'Haltausfälle'
]
delay_keywords = [
    'Verspätung', 'verspäten', 'verspäten sich', 'verspätet', 'später', 'Wartezeit',
    'warten', 'spätere', 'verlängert'
]
delay_exact_keywords = [
    'Folgeverspätung', 'Folgeverspätungen', 'Verspätungskürzung', 'spätere Fahrzeiten', 'Folgeverzögerung',
    'spätere Ankunft', 'verlängerte Fahrzeit', 'Verzögerung', 'verzögert', 'Verzögerungen', 'verzögern',
    'Folgeverzögerungen'
]
delay_lower_priority_keywords = [
    '#Störung', 'Technische Störung', 'Technische_Störung', 'Fahrplanabweichung', 'unregelmäßiger Zugverkehr',
    'unregelmäßigem Zugverkehr', 'unregelmäßigen Zugverkehr', 'Störung', 'Störungsinformation',
]
obstruction_keywords = [
    'Umleitung', 'Umleitungen', 'umgeleitet', 'Sperrung', 'gesperrt', 'Vollsperrung', 'Verkehrsbehinderung',
    'Behinderung', 'behindern', 'blockiert', 'Blockade', 'unterbrochen', 'Behinderungen', 'Beeinträchtigung',
    'beeinträchtigt', 'beeinträchtigen', 'voll gesperrt', 'Verkehrsbehinderungen', 'Einschränkungen'
]
obstruction_lower_priority_keywords = [
    'Großbaustelle', 'Nachtbaustelle', 'Tagesbaustelle', 'Baustelle', 'Bauarbeiten', 'Straßenbauarbeiten',
    'Straßenbau', 'Fliegerbombe', 'Fliegerbomben', 'Bombenentschärfung', 'Schwertransport', 'Vorsicht',
    'brennender PKW', 'Störung'
]
railreplacementservice_keywords = [
    'Schienenersatzverkehr', '#SEV', 'Ersatzverkehr', 'Pendelverkehr', 'ersetzt', 'ersetzen', 'Ersatz', 'SEV'
]
railreplacementservice_exact_keywords = [
    'Busnotverkehr', 'durch Busse ersetzt', 'Ersatzzug', 'Ersatzbus', 'Bus ersetzt', 'Bus statt Bahn', 'Ersatzbusse',
    'Ersatz durch Busse', 'durch einen Bus ersetzt', 'SEV mit Bussen', 'Busse statt S-Bahnen'
]
trafficjam_keywords = [
    'Stau', 'Staus'
]
trafficjam_exact_keywords = [
    'Blechlawine', 'Blechlawinen', 'Autoschlange', 'Stauung', 'Stop and Go', 'zähfließender Verkehr',
    'Verkehrsüberlastung', 'Rückstau', 'Verkehrsüberlastung', 'Hohes Verkehrsaufkommen', 'stockender Verkehr',
    'lahmender Verkehr', 'staut'
]


@labeling_function(pre=[])
def lf_accident_chained(x):
    if any(accident_lf(x) == Accident for accident_lf in
           [
               lf_accident,
               lf_accident_street,
               lf_accident_no_cause_check
           ]):
        return Accident
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_accident(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], accident_keywords)
    highest_lower_priority = process.extractOne(x.trigger['text'], accident_lower_priority_keywords)
    if highest[1] >= 90 or highest_lower_priority[1] > 90:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        else:
            return Accident
    return ABSTAIN


@labeling_function(pre=[])
def lf_accident_street(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], accident_keywords)
    highest_lower_priority = process.extractOne(x.trigger['text'], accident_lower_priority_keywords)
    if highest[1] >= 90 or highest_lower_priority[1] > 90 or x.trigger['text'] in accident_exact_keywords:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif 'location_street' in x.entity_type_freqs:
            return Accident
    return ABSTAIN


@labeling_function(pre=[])
def lf_accident_no_cause_check(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], accident_keywords)
    if highest[1] >= 90:
        if check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        else:
            return Accident
    return ABSTAIN


def check_cause_keywords(left_tokens, x):
    """
    Checks the tokens (usually the ones left to the trigger) for causal keywords that would
    indicate that the trigger is not the event of the sentence, but rather a cause for the actual event.
    :param x: DataPoint.
    :param left_tokens: Context tokens of the trigger.
    :return: True or False depending on a match with any of the causal keywords.
    """
    cause_keywords = ['nach', 'wegen', 'grund', 'aufgrund', 'durch']
    cause_token_idx, cause_word = next(((idx, token) for idx, token in enumerate(left_tokens)
                                        if token.lower() in cause_keywords), (-1, None))
    if cause_token_idx > -1:
        # Avoid cases such as: 'durch Busse ersetzt'
        if cause_word == 'durch':
            highest = process.extractOne(x.trigger['text'], railreplacementservice_keywords)
            if highest[1] >= 90 and 'location_route' in x.entity_type_freqs:
                return False
        # make sure that no other entity occurs after the causal keyword or is the one containing the cause_keyword
        if cause_token_idx == len(left_tokens) - 1:
            # adjacent to trigger
            return True
        for entity in x.entities:
            cause_token_idx_global = x.trigger['start'] - len(left_tokens) + cause_token_idx
            if entity['id'] != x.trigger['id'] and cause_token_idx_global < entity['end'] <= x.trigger['start']:
                return False
        return True
    else:
        return False


def check_in_parentheses(trigger_text, left_tokens=None, right_tokens=None):
    """
    Checks if the trigger text contains parentheses (is surrounded by parentheses),
    which indicates that the trigger is not the event in the sentence.
    :param trigger_text: Trigger text.
    :param left_tokens: Tokens left of the trigger.
    :param right_tokens: Tokens right of the trigger.
    :return: True or False depending on whether the trigger text contains parentheses.
    """
    parentheses = re.compile('\\(.*\\)')
    if parentheses.match(trigger_text):
        return True
    elif left_tokens and right_tokens:
        if parentheses.match(''.join([left_tokens[-1], trigger_text, right_tokens[0]])):
            return True
    else:
        return False


@labeling_function(pre=[])
def lf_canceledroute_cat(x):
    """
    Checks for canceled route keywords.
    :param x:
    :return:
    """
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], canceledroute_keywords)
    highest_exact = process.extractOne(x.trigger['text'], canceledroute_exact_keywords)
    if (highest[1] >= 90 or highest_exact[1] > 90) and 'location_route' in x.entity_type_freqs:
        if x.trigger['text'] in ['aus', 'aus.'] and any(fall in trigger_left_tokens for fall in ['fällt', 'fallen']):
            return ABSTAIN
        return CanceledRoute
    return ABSTAIN


@labeling_function(pre=[])
def lf_canceledroute_replicated(x):
    """
    Replicated canceled route function. Temporary solution to assign canceled route labeling function more importance.
    :param x:
    :return:
    """
    return lf_canceledroute_cat(x)


@labeling_function(pre=[])
def lf_canceledstop_cat(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], canceledstop_keywords)
    highest_exact = process.extractOne(x.trigger['text'], canceledstop_exact_keywords)
    if (highest[1] >= 90 or highest_exact[1] > 90) and 'location_stop' in x.entity_type_freqs:
        if not check_route_keywords(trigger_left_tokens[-7:]):
            return CanceledStop
    return ABSTAIN


@labeling_function(pre=[])
def lf_canceledstop_replicated(x):
    """
    Replicated canceled stop function. Temporary solution to assign canceled stop labeling function more importance.
    :param x:
    :return:
    """
    return lf_canceledstop_cat(x)


def check_route_keywords(tokens):
    route_keywords = ['Strecke', 'Streckenabschnitt', 'Abschnitt', 'Linie', 'zwischen']
    if any(route_keyword in tokens for route_keyword in route_keywords):
        return True
    else:
        return False


@labeling_function(pre=[])
def lf_delay_chained(x):
    if any(delay_lf(x) == Delay for delay_lf in
           [
               lf_delay_cat,
               lf_delay_duration,
               lf_delay_priorities
           ]):
        return Delay
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_delay_cat(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], delay_keywords)
    highest_exact = process.extractOne(x.trigger['text'], delay_exact_keywords)
    if highest[1] >= 90 or highest_exact[1] > 90:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif is_negated(trigger_left_tokens):
            return ABSTAIN
        else:
            return Delay
    return ABSTAIN


@labeling_function(pre=[pre_closest_entity_same_sentence])
def lf_delay_duration_check(x):
    if x['closest_entity'] and x['closest_entity']['entity_type'] == 'duration':
        return Delay
    else:
        return ABSTAIN


@labeling_function(pre=[pre_closest_entity_same_sentence])
def lf_delay_duration_positional_check(x):
    if x['closest_entity'] and x['closest_entity']['entity_type'] == 'duration' and \
            x['closest_entity']['start'] < x['trigger']['start']:
        return Delay
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_delay_duration(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], delay_keywords)
    highest_exact = process.extractOne(x.trigger['text'], delay_exact_keywords)
    if highest[1] >= 90 or highest_exact[1] > 90:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif is_negated(trigger_left_tokens):
            return ABSTAIN
        elif 'duration' in x.entity_type_freqs:
            return Delay
    return ABSTAIN


@labeling_function(pre=[])
def lf_delay_priorities(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], delay_keywords)
    highest_lower_priority = process.extractOne(x.trigger['text'], delay_lower_priority_keywords)
    highest_exact = process.extractOne(x.trigger['text'], delay_exact_keywords)
    if highest[1] >= 90 or highest_exact[1] > 90 or highest_lower_priority[1] >= 90:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif is_negated(trigger_left_tokens):
            return ABSTAIN
        elif x.entity_type_freqs['trigger'] > 1 and highest_lower_priority[1] >= 90:
            # Check for other higher priority delay trigger: "Verspätung" vs. lower priority "Störung"
            higher_priority_delay = False
            for entity in x.entities:
                if entity['entity_type'] == 'trigger' and entity['id'] != x.trigger['id']:
                    best_match = process.extractOne(entity['text'], delay_keywords)
                    best_exact_match = process.extractOne(entity['text'], delay_exact_keywords)
                    if best_match[1] >= 90 or best_exact_match[1] > 90:
                        higher_priority_delay = True
            if higher_priority_delay:
                return ABSTAIN
            else:
                return Delay
        else:
            return Delay
    return ABSTAIN


def is_negated(trigger_left_tokens):
    return trigger_left_tokens and trigger_left_tokens[-1] in ['kein', 'keine', 'keinen', 'ohne']


@labeling_function(pre=[])
def lf_obstruction_chained(x):
    if any(obstruction_lf(x) == Obstruction for obstruction_lf in
           [
               lf_obstruction_cat,
               lf_obstruction_street,
               lf_obstruction_priorities
           ]):
        return Obstruction
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_obstruction_cat(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], obstruction_keywords)
    if highest[1] >= 90 and x.trigger['text'] not in ['aus', 'aus.']:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif is_negated(trigger_left_tokens):
            return ABSTAIN
        else:
            return Obstruction
    return ABSTAIN


@labeling_function(pre=[])
def lf_obstruction_street(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], obstruction_keywords)
    highest_lower_priority = process.extractOne(x.trigger['text'], obstruction_lower_priority_keywords)
    if (highest[1] >= 90 or highest_lower_priority[1] >= 90) and x.trigger['text'] not in ['aus', 'aus.']:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif is_negated(trigger_left_tokens):
            return ABSTAIN
        elif 'location_street' in x.entity_type_freqs:
            return Obstruction
    return ABSTAIN


@labeling_function(pre=[])
def lf_obstruction_priorities(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], obstruction_keywords)
    highest_lower_priority = process.extractOne(x.trigger['text'], obstruction_lower_priority_keywords)
    if (highest[1] >= 90 or highest_lower_priority[1] >= 90) and x.trigger['text'] not in ['aus', 'aus.']:
        if (check_cause_keywords(trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif is_negated(trigger_left_tokens):
            return ABSTAIN
        elif x.entity_type_freqs['trigger'] > 1 and highest_lower_priority[1] >= 90:
            # Check for other higher priority obstruction trigger: "Sperrung" vs. lower priority "Baustelle"
            higher_priority_obstruction = False
            for entity in x.entities:
                if entity['entity_type'] == 'trigger' and entity['id'] != x.trigger['id']:
                    best_match = process.extractOne(entity['text'], obstruction_keywords)
                    if best_match[1] >= 90:
                        higher_priority_obstruction = True
            if higher_priority_obstruction:
                return ABSTAIN
            else:
                return Obstruction
        else:
            return Obstruction
    return ABSTAIN


@labeling_function(pre=[])
def lf_obstruction_negative(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], obstruction_keywords)
    highest_lower_priority = process.extractOne(x.trigger['text'], obstruction_lower_priority_keywords)
    if (highest[1] >= 90 or highest_lower_priority[1] >= 90) and x.trigger['text'] not in ['aus', 'aus.']:
        if is_negated(trigger_left_tokens):
            return O
    return ABSTAIN


@labeling_function(pre=[])
def lf_railreplacementservice_cat(x):
    highest = process.extractOne(x.trigger['text'], railreplacementservice_keywords)
    highest_exact = process.extractOne(x.trigger['text'], railreplacementservice_exact_keywords)
    if (highest[1] >= 90 or highest_exact[1] > 90) and 'location_route' in x.entity_type_freqs:
        return RailReplacementService
    return ABSTAIN


@labeling_function(pre=[])
def lf_railreplacementservice_replicated(x):
    """
    Replicated rail replacement service function. Temporary solution to assign rrs labeling function more importance.
    :param x:
    :return:
    """
    return lf_railreplacementservice_cat(x)


@labeling_function(pre=[])
def lf_trafficjam_chained(x):
    if any(trafficjam_lf(x) == TrafficJam for trafficjam_lf in
           [
               lf_trafficjam_cat,
               lf_trafficjam_street,
               lf_trafficjam_order
           ]):
        return TrafficJam
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_trafficjam_cat(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], trafficjam_keywords)
    highest_exact = process.extractOne(x.trigger['text'], trafficjam_exact_keywords)
    if (highest[1] >= 90 or highest_exact[1] > 90) and \
            x.trigger['text'] not in ['aus', 'aus.']:
        if check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens) and \
                x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        else:
            return TrafficJam
    return ABSTAIN


@labeling_function(pre=[pre_closest_entity_same_sentence])
def lf_trafficjam_distance_check(x):
    if x['closest_entity'] and x['closest_entity']['entity_type'] == 'distance':
        return TrafficJam
    else:
        return ABSTAIN


@labeling_function(pre=[pre_closest_entity_same_sentence])
def lf_trafficjam_distance_positional_check(x):
    if x['closest_entity'] and x['closest_entity']['entity_type'] == 'distance' and \
            x['closest_entity']['start'] < x['trigger']['start']:
        return TrafficJam
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_trafficjam_street(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], trafficjam_keywords)
    highest_exact = process.extractOne(x.trigger['text'], trafficjam_exact_keywords)
    if (highest[1] >= 90 or highest_exact[1] > 90) and \
            x.trigger['text'] not in ['aus', 'aus.']:
        if check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens) and \
                x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif 'location_street' in x.entity_type_freqs:
            return TrafficJam
    return ABSTAIN


@labeling_function(pre=[])
def lf_trafficjam_order(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    trigger_right_tokens = get_windowed_right_tokens(x.trigger, x.tokens)
    highest = process.extractOne(x.trigger['text'], trafficjam_keywords)
    highest_exact = process.extractOne(x.trigger['text'], trafficjam_exact_keywords)
    if (highest[1] >= 90 or highest_exact[1] > 90) and \
            x.trigger['text'] not in ['aus', 'aus.']:
        if check_in_parentheses(x.trigger['text'], trigger_left_tokens, trigger_right_tokens) and \
                x.entity_type_freqs['trigger'] > 1:
            return ABSTAIN
        elif x.entity_type_freqs['trigger'] > 1:
            not_first_trafficjam_trigger = False
            for entity in x.entities:
                if entity['start'] < x.trigger['start'] and entity['entity_type'] == 'trigger' \
                        and entity['id'] != x.trigger['id']:
                    best_match = process.extractOne(entity['text'], trafficjam_keywords)
                    best_exact_match = process.extractOne(entity['text'], trafficjam_exact_keywords)
                    if best_match[1] >= 90 or best_exact_match[1] > 90:
                        not_first_trafficjam_trigger = True
            if not_first_trafficjam_trigger:
                return ABSTAIN
            else:
                return TrafficJam
        else:
            return TrafficJam
    return ABSTAIN


@labeling_function(pre=[])
def lf_negative(x):
    """
    Simple negative labeling function that returns the negative trigger label when all other labeling functions abstain.
    This may hinder the generalization power of the model.
    :param x:
    :return:
    """
    lfs = [
        lf_accident_chained,
        lf_canceledroute_cat,
        lf_canceledstop_cat,
        lf_delay_chained,
        lf_obstruction_chained,
        lf_railreplacementservice_cat,
        lf_trafficjam_chained
    ]
    for lf in lfs:
        if lf(x) != ABSTAIN:
            if lf(x) == O:
                return O
            else:
                return ABSTAIN
    return O


@labeling_function(pre=[])
def lf_cause_negative(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    if check_cause_keywords(trigger_left_tokens[-4:], x) and x.entity_type_freqs['trigger'] > 1:
        return O
    else:
        return ABSTAIN


@labeling_function(pre=[])
def lf_negated_event(x):
    trigger_left_tokens = get_windowed_left_tokens(x.trigger, x.tokens)
    if is_negated(trigger_left_tokens):
        return O
    else:
        return ABSTAIN
