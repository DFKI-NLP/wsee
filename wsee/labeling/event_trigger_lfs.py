from snorkel.labeling import labeling_function
from snorkel.labeling.lf.nlp import nlp_labeling_function
from fuzzywuzzy import process
from wsee.preprocessors.preprocessors import get_trigger, get_left_tokens, get_right_tokens, get_entity_type_freqs


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
        'Streckensperrung'
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
        'blockiert', 'Blockade', 'unterbrochen'
]
railreplacementservice_keywords = [
        'Schienenersatzverkehr', '#SEV', 'Ersatzverkehr', 'Busnotverkehr', 'Pendelverkehr', 'durch Busse ersetzt',
        'Ersatzzug', 'Ersatzbus'
]
trafficjam_keywords = [
        'Stau', 'Staus', 'Staumeldung', 'Stauwarnung', 'Blechlawine', 'Blechlawinen', 'stockender Verkehr',
        '#stautweet', 'Verkehrsüberlastung', 'Rückstau', 'Verkehrsüberlastung', 'Hohes Verkehrsaufkommen'
]


@labeling_function(pre=[get_trigger])
def lf_accident_cat(x):
    highest = process.extractOne(x.trigger['text'], accident_keywords)
    if highest[1] > 90:
        return Accident
    return ABSTAIN


@labeling_function(pre=[get_trigger])
def lf_canceledroute_cat(x):
    highest = process.extractOne(x.trigger['text'], canceledroute_keywords)
    if highest[1] > 90:
        return CanceledRoute
    return ABSTAIN


@labeling_function(pre=[get_trigger])
def lf_canceledstop_cat(x):
    highest = process.extractOne(x.trigger['text'], canceledstop_keywords)
    if highest[1] > 90:
        return CanceledStop
    return ABSTAIN


@labeling_function(pre=[get_trigger])
def lf_delay_cat(x):
    highest = process.extractOne(x.trigger['text'], delay_keywords)
    if highest[1] > 90:
        return Delay
    return ABSTAIN


@labeling_function(pre=[get_trigger])
def lf_obstruction_cat(x):
    highest = process.extractOne(x.trigger['text'], obstruction_keywords)
    if highest[1] > 90:
        return Obstruction
    return ABSTAIN


@labeling_function(pre=[get_trigger])
def lf_railreplacementservice_cat(x):
    highest = process.extractOne(x.trigger['text'], railreplacementservice_keywords)
    if highest[1] > 90:
        return RailReplacementService
    return ABSTAIN


@labeling_function(pre=[get_trigger])
def lf_trafficjam_cat(x):
    highest = process.extractOne(x.trigger['text'], trafficjam_keywords)
    if highest[1] > 90:
        return TrafficJam
    return ABSTAIN
