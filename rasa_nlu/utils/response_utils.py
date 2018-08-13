__author__ = 'xueyu'

import logging
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message

logger = logging.getLogger(__name__)


class ResponseComponent(Component):

    name = "response_component"

    def process(self, message, **kwargs):

        print(message.as_dict())

        intent = message.get('intent')
        intent_ranking = message.get('intent_ranking')
        entities = message.get('entities')
        slots = []
        if entities:
            for entitie in entities:
                slot = {}
                slot['confidence'] = entitie.get('confidence', None)
                slot["name"] = entitie.get("entity", None)
                slot["original_word"] = entitie.get('value', None)
                slot["normalized_word"] = entitie.get('value', None)
                slot["begin"] = entitie.get('start', 0)
                end = entitie.get('end', 0)
                slot["length"] = end - slot["begin"]

                slots.append(slot)

        schema = {
            'domain_confidence': 0,
            'intent': intent['name'],
            'intent_confidence': intent['confidence'],
            'slots': slots
        }

        message.set('schema', schema, add_to_output=True)

        action_list = [
            {
                "action_id": "",
                "refine_detail": {
                    "option_list": [],
                    "interact": "",
                    "clarify_reason": ""
                },
                "confidence": 0,
                "custom_reply": "",
                "say": "",
                "type": "understood"
            }
        ]

        message.set('action_list', action_list, add_to_output=True)

