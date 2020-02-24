NEGATIVE_TRIGGER_LABEL = 'O'
NEGATIVE_ARGUMENT_LABEL = 'no_arg'

SD4M_RELATION_TYPES = ['Accident', 'CanceledRoute', 'CanceledStop', 'Delay',
                       'Obstruction', 'RailReplacementService', 'TrafficJam',
                       NEGATIVE_TRIGGER_LABEL]
# if we use their indices (-1), we might want to move 'Other' to the beginning
ROLE_LABELS = ['location', 'delay', 'direction',
               'start_loc', 'end_loc',
               'start_date', 'end_date', 'cause',
               'jam_length', 'route', NEGATIVE_ARGUMENT_LABEL]
