from enum import Enum
from typing import Optional
import purepyindi2
from dataclasses import dataclass
import xml.etree.ElementTree as ET

DEFAULT_DEBOUNCE_SEC = 3

class Operation(Enum):
    EQ = 'eq'
    LT = 'lt'
    LE = 'le'
    GT = 'gt'
    GE = 'ge'
    NE = 'ne'
    BETWEEN = 'between'
    def __str__(self):
        return self.value

@dataclass(eq=True, frozen=True)
class Transition:
    value : Optional[purepyindi2.AnyIndiValue]
    value_2 : Optional[purepyindi2.AnyIndiValue]
    debounce_sec : float = DEFAULT_DEBOUNCE_SEC
    op : Optional[Operation] = None

    def compare(self, new_value):
        if self.op is None:
            return True
        if self.op is Operation.EQ:
            return new_value == self.value
        elif self.op is Operation.NE:
            return new_value != self.value
        else:
            try:
                new_value = float(new_value)
            except (ValueError, TypeError):
                return False
            if self.op is Operation.LT:
                return new_value < self.value
            elif self.op is Operation.LE:
                return new_value <= self.value
            elif self.op is Operation.GT:
                return new_value > self.value
            elif self.op is Operation.GE:
                return new_value >= self.value
            elif self.op is Operation.BETWEEN:
                lo = min(self.value, self.value_2)
                hi = max(self.value, self.value_2)
                return lo <= new_value < hi
        return False

@dataclass
class Reaction:
    indi_id : str
    transitions : dict[Transition, list[str]]

@dataclass
class Personality:
    reactions : list[Reaction]
    default_voice : str
    random_utterances : list[str]
    
    @classmethod
    def from_path(cls, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        reactions = []
        random_utterances = []
        default_voice = None

        for el in root:
            transitions = {}
            if el.tag == 'default-voice':
                default_voice = el.attrib['name']
                continue
            elif el.tag == 'random-utterances':
                for utterance in el:
                    random_utterances.append(ET.tostring(utterance, 'utf-8').decode('utf8').strip())
                continue
            assert el.tag == 'react-to'
            indi_id = el.attrib['indi-id']
            for transition in el:
                assert transition.tag == 'transition'
                if 'low' in transition.attrib:
                    value = purepyindi2.parse_string_into_any_indi_value(transition.attrib['low'])
                    value_2 = purepyindi2.parse_string_into_any_indi_value(transition.attrib['high'])
                    operation = Operation.BETWEEN
                elif 'value' in transition.attrib:
                    value = purepyindi2.parse_string_into_any_indi_value(transition.attrib['value'])
                    value_2 = None
                    operation = purepyindi2.parse_string_into_enum(transition.attrib.get('op', 'eq'), Operation)
                else:
                    value = None
                    value_2 = None
                    operation = None
                if 'debounce_sec' in transition.attrib:
                    debounce_sec = float(transition.attrib['debounce_sec'])
                else:
                    debounce_sec = DEFAULT_DEBOUNCE_SEC
                trans = Transition(op=operation, value=value, value_2=value_2, debounce_sec=debounce_sec)
                if trans in transitions:
                    raise RuntimeError(f"Multiply defined for {indi_id} {operation=} {value=}")
                transitions[trans] = []
                for utterance in transition:
                    assert utterance.tag == 'speak'
                    transitions[trans].append(ET.tostring(utterance, 'utf-8').decode('utf8').strip())
            reactions.append(Reaction(indi_id=indi_id, transitions=transitions))
        return cls(reactions=reactions, default_voice=default_voice, random_utterances=random_utterances)

if __name__ == "__main__":
    import pprint
    pprint.pprint(Personality.from_path('./default.xml'), width=255)