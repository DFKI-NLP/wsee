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
        'Fahrzeugstörung', 'Bahnübergangsstörung', 'Stellwerksausfall', 'Teilausfall', 'Teilausfällen',  'Ampelstörung',
        'Triebwagenstörung'
]
intervention_keywords = [
        'Intervention', 'Notarzteinsatz', 'Notarzt', 'Notarzteinsatzes', 'Polizeieinsatz',
        'Polizeieinsatzes', 'Feuerwehreinsatz', 'Feuerwehreinsatzes'
]
accident_keywords = [
        'Unfall', 'Unfälle', 'Verkehrsunfall', 'Verkehrsunfälle', 'Autounfall', 'Autounfälle', 'Massenkarambolage',
        'Auffahrunfall', 'Zusammenstoß', 'Geisterfahrer', 'Falschfahrer', 'Unfallstelle', 'Zugkollision',
        'Zugkollisionen', 'Zugunglück', 'Personenschaden', 'Bergungsarbeit', 'Bergungsarbeiten', 'Unfalles',
        'Verkehrsunfalls'
]
canceledroute_keywords = [
        'Zugausfall', 'Zugausfälle', 'Zugausfällen', 'S-Bahn-Ausfall', 'fällt aus', 'ausgefallen', 'gesperrt',
        'Umleitung', 'umgeleitet', 'Einstellung', 'Teilausfall', 'Ausfall', 'streichen', 'gestrichen',
        'Streckensperrung',
        'aus', 'fällt', 'fallen', 'unterbrochen', 'Ausfälle', 'Einstellung'
]
canceledstop_keywords = [
        'hält nicht', 'halten nicht', 'hält', 'geschlossen', 'gesperrt'
]
delay_keywords = [
        'Verspätung', 'Verspätungen', 'Verzögerung', 'Verzögerungen', '#Störung',
        'Technische Störung', 'Technische_Störung', 'Folgeverspätung', 'Folgeverspätungen',
        'Fahrplanabweichung', 'unregelmäßiger Zugverkehr', 'unregelmäßigem Zugverkehr', 'unregelmäßigen Zugverkehr',
        'Verspätungskürzung', 'verspäten', 'verspäten sich', 'Störung', 'Störungsinformation', 'verzögert',
        'verspätet', 'später', 'Wartezeit', 'warten'
]
obstruction_keywords = [
        'Umleitung', 'Umleitungen', 'umgeleitet',  'Sperrung', 'gesperrt', 'Großbaustelle', 'Nachtbaustelle',
        'Tagesbaustelle', 'Baustelle', 'Bauarbeiten', 'Straßenbauarbeiten', 'Straßenbau', 'Fliegerbombe',
        'Fliegerbomben', 'Bombenentschärfung', 'Vollsperrung', 'Verkehrsbehinderung', 'Behinderung', 'behindern',
        'blockiert', 'Blockade', 'unterbrochen',
        'Behinderungen', 'Schwertransport', 'Vorsicht', 'brennender PKW', 'Störung'
]
railreplacementservice_keywords = [
        'Schienenersatzverkehr', '#SEV', 'Ersatzverkehr', 'Busnotverkehr', 'Pendelverkehr', 'durch Busse ersetzt',
        'Ersatzzug', 'Ersatzbus', 'Bus ersetzt'
]
trafficjam_keywords = [
        'Stau', 'Staus', 'Staumeldung', 'Stauwarnung', 'stockender Verkehr',
        '#stautweet', 'Verkehrsüberlastung', 'Rückstau', 'Verkehrsüberlastung', 'Hohes Verkehrsaufkommen'
]
trafficjam_exact_keywords = [
        'Blechlawine', 'Blechlawinen'
]


@labeling_function(pre=[get_trigger])
def lf_accident_cat(x):
    highest = process.extractOne(x.trigger['text'], accident_keywords)
    if highest[1] >= 90:
        return Accident
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_entity_type_freqs, get_trigger_left_tokens, get_trigger_right_tokens])
def lf_accident_context(x):
    highest = process.extractOne(x.trigger['text'], accident_keywords)
    if highest[1] >= 90:
        if (check_cause_keywords(x.trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], x.trigger_left_tokens, x.trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return O
        else:
            return Accident
    return ABSTAIN


def check_cause_keywords(tokens, x):
    """
    Checks the tokens (usually the ones left to the trigger) for causal keywords that would
    indicate that the trigger is not the event of the sentence, but rather a cause for the actual event.
    :param x: DataPoint.
    :param tokens: Context tokens of the trigger.
    :return: True or False depending on a match with any of the causal keywords.
    """
    cause_keywords = ['nach', 'wegen', 'bei', 'grund', 'aufgrund', 'durch']
    if any(token.lower() in cause_keywords for token in tokens):
        left_text = " ".join(tokens)
        lower_left_text = left_text.lower()
        cause_start = max([lower_left_text.find(cause_keyword) for cause_keyword in cause_keywords])
        assert cause_start > -1
        # make sure that no other trigger occurs after the causal keyword
        if x.event_triggers:
            rightest_trigger_start = max([left_text.find(get_entity(event_trigger['id'], x.entities)['text'])
                                          for event_trigger in x.event_triggers])
            if rightest_trigger_start > cause_start:
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


@labeling_function(pre=[get_trigger, get_trigger_right_tokens, get_entity_type_freqs])
def lf_canceledroute_cat(x):
    """
    Checks for canceled route keywords. Does not handle special case of split trigger "fällt ... aus".
    The annotators only annotated one of the two words as an event.
    :param x:
    :return:
    """
    highest = process.extractOne(x.trigger['text'], canceledroute_keywords)
    if highest[1] >= 90 and 'location_route' in x.entity_type_freqs:
        # TODO handle special case?
        # if x.trigger['text'] in ['fällt', 'fallen'] and 'aus' in x.trigger_right_tokens:
        # if x.trigger['text'] == 'aus' and any(fall in x.trigger_left_tokens for fall in ['fällt', 'fallen']):
        #    return ABSTAIN
        return CanceledRoute
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_entity_type_freqs, get_trigger_left_tokens])
def lf_canceledstop_cat(x):
    highest = process.extractOne(x.trigger['text'], canceledstop_keywords)
    if highest[1] >= 90 and 'location_stop' in x.entity_type_freqs:
        if not check_route_keywords(x.trigger_left_tokens[-7:]):
            return CanceledStop
    return ABSTAIN


def check_route_keywords(tokens):
    route_keywords = ['Strecke', 'Streckenabschnitt', 'Abschnitt', 'Linie', 'zwischen']
    if any(route_keyword in tokens for route_keyword in route_keywords):
        return True
    else:
        return False


@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs])
def lf_delay_cat(x):
    highest = process.extractOne(x.trigger['text'], delay_keywords)
    if highest[1] >= 90:
        if (check_cause_keywords(x.trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], x.trigger_left_tokens, x.trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return O
        else:
            return Delay
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs])
def lf_obstruction_cat(x):
    highest = process.extractOne(x.trigger['text'], obstruction_keywords)
    if highest[1] >= 90 and x.trigger['text'] not in ['aus', 'aus.']:
        if (check_cause_keywords(x.trigger_left_tokens[-4:], x) or
            check_in_parentheses(x.trigger['text'], x.trigger_left_tokens, x.trigger_right_tokens)) \
                and x.entity_type_freqs['trigger'] > 1:
            return O
        else:
            return Obstruction
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_entity_type_freqs])
def lf_railreplacementservice_cat(x):
    highest = process.extractOne(x.trigger['text'], railreplacementservice_keywords)
    if highest[1] >= 90 and 'location_route' in x.entity_type_freqs:
        return RailReplacementService
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs])
def lf_trafficjam_cat(x):
    highest = process.extractOne(x.trigger['text'], trafficjam_keywords)
    if (highest[1] >= 90 or x.trigger['text'] in trafficjam_exact_keywords) and \
            x.trigger['text'] not in ['aus', 'aus.']:
        if check_in_parentheses(x.trigger['text'], x.trigger_left_tokens, x.trigger_right_tokens) and \
                x.entity_type_freqs['trigger'] > 1:
            return O
        else:
            return TrafficJam
    return ABSTAIN


@labeling_function(pre=[get_trigger, get_trigger_left_tokens, get_trigger_right_tokens, get_entity_type_freqs])
def lf_negative(x):
    lfs = [
        lf_accident_context,
        lf_canceledroute_cat,
        lf_canceledstop_cat,
        lf_delay_cat,
        lf_obstruction_cat,
        lf_railreplacementservice_cat,
        lf_trafficjam_cat
    ]
    for lf in lfs:
        if lf(x) != ABSTAIN:
            if lf(x) == O:
                return O
            else:
                return ABSTAIN
    return O
