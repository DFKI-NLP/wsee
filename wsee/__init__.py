from wsee.dataset_readers import *
from wsee.models import *
from wsee.predictors import *
from wsee.utils import *

NEGATIVE_TRIGGER_LABEL = 'O'
NEGATIVE_ARGUMENT_LABEL = 'no_arg'

SD4M_RELATION_TYPES = ['Accident', 'CanceledRoute', 'CanceledStop', 'Delay',
                       'Obstruction', 'RailReplacementService', 'TrafficJam',
                       NEGATIVE_TRIGGER_LABEL]

SDW_RELATION_TYPES = ["OrganizationLeadership", "Acquisition",
                      "Disaster", "Insolvency", "Layoffs", "Merger", "SpinOff", "Strike",
                      "CompanyProvidesProduct", "CompanyUsesProduct", "CompanyTurnover",
                      "CompanyRelationship", "CompanyFacility", "CompanyIndustry",
                      "CompanyHeadquarters", "CompanyWebsite", "CompanyWikipediaSite",
                      "CompanyNumEmployees", "CompanyCustomer", "CompanyProject",
                      "CompanyFoundation", "CompanyTermination", "CompanyFinancialEvent"]

# if we use their indices (-1), we might want to move 'Other' to the beginning
ROLE_LABELS = ['location', 'delay', 'direction',
               'start_loc', 'end_loc',
               'start_date', 'end_date', 'cause',
               'jam_length', 'route', NEGATIVE_ARGUMENT_LABEL]
