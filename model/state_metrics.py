"""Lexical state-marker checks, not semantic or human consistency ratings."""
import re
from protocol import parse_state

def slot_metrics(predictions, rows, domain_rows):
    if len(predictions) != len(rows) or not rows:
        raise ValueError('Expected one prediction per nonempty evaluation row')
    scores = {}
    for field in ('NPC', 'Player'):
        vocabulary = {parse_state(r['input'])[field] for r in domain_rows}
        correct = []
        for prediction, row in zip(predictions, rows):
            desired = parse_state(row['input'])[field]
            mentioned = {value for value in vocabulary | {desired}
                         if re.search(r'\b'+re.escape(value)+r'\b', prediction.lower())}
            correct.append(mentioned == {desired})
        scores[field.lower()+'_explicit_accuracy'] = sum(correct)/len(correct)
    # Additional markers are specific to the frozen synthetic generator, not
    # general semantic consistency rules or ratings for arbitrary narratives.
    contexts = {
        'Location': {'forest':'forest trail','village':'village square','cave':'cave entrance','river':'riverbank'},
        'Time': {'morning':'in the morning','night':'by lantern light'},
        'Weather': {'clear':'under clear skies','rainy':'beneath a travel canopy'},
    }
    if all(r.get('source') == 'rule_template_ai_reviewed' for r in rows):
        for field, mapping in contexts.items():
            matches = []
            for prediction, row in zip(predictions, rows):
                desired = parse_state(row['input'])[field]
                if desired not in mapping:
                    break
                mentioned = {value for value, phrase in mapping.items() if phrase in prediction.lower()}
                matches.append(mentioned == {desired})
            if len(matches) == len(rows):
                scores[field.lower()+'_context_marker_accuracy'] = sum(matches)/len(matches)
    return scores
