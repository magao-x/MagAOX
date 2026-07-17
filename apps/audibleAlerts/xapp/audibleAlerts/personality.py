from dataclasses import dataclass
from enum import Enum
import os.path
import logging
import pathlib
from typing import Optional
import tomllib

import purepyindi2

from magaox.tts.core import PlaybackRequest, Recording, DynamicSpeech, Voice, load_voice

log = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_SEC = 3

def action_dict_to_playback_request(action_dict):
    print(f"{action_dict=}")
    if action_dict['type'] == 'play':
        return Recording(action_dict['file'])
    elif action_dict['type'] == 'speak':
        return DynamicSpeech(action_dict['text'], custom_voice_name=action_dict.get('voice'))

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
    # id duplicated here from Reaction because Transition is used as a dict key and we need uniqueness:
    indi_id : str
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
    transitions : dict[Transition, list[PlaybackRequest]]

@dataclass
class Personality:
    reactions : list[Reaction]
    default_voice : Voice
    random_utterances : list[PlaybackRequest]
    soundboard : dict[str, PlaybackRequest]
    walkups : dict[str, list[PlaybackRequest]]

    @classmethod
    def from_path(cls, file_path: pathlib.Path):
        reactions = []
        random_utterances = []
        soundboard = {}
        default_voice = None
        walkups = {}
        with file_path.open('rb') as fh:
            root = tomllib.load(fh)

        default_voice = load_voice(root['default_voice'])
        for utterance in root['random_utterances']:
            random_utterances.append(action_dict_to_playback_request(utterance))
        for btn in root['soundboard']:
            soundboard[btn] = action_dict_to_playback_request(root['soundboard'][btn])

        for wup_email in root['walkups']:
            email = wup_email.replace('.', '-dot-').replace('@', '-at-')
            walkups[email] = []
            for walkup_clip in root['walkups'][wup_email]:
                walkups[email].append(Recording(walkup_clip))

        

        for react in root['react']:
            indi_id = react['indi_id']
            transitions = {}
            for transition in react['transition']:
                if 'low' in transition:
                    value = purepyindi2.parse_string_into_any_indi_value(transition['low'])
                    value_2 = purepyindi2.parse_string_into_any_indi_value(transition['high'])
                    operation = Operation.BETWEEN
                elif 'value' in transition:
                    value = purepyindi2.parse_string_into_any_indi_value(transition['value'])
                    value_2 = None
                    operation = purepyindi2.parse_string_into_enum(transition.get('op', 'eq'), Operation)
                else:
                    value = None
                    value_2 = None
                    operation = None
                debounce_sec = DEFAULT_DEBOUNCE_SEC
                trans = Transition(indi_id=indi_id, op=operation, value=value, value_2=value_2, debounce_sec=debounce_sec)
                if trans in transitions:
                    raise RuntimeError(f"Multiply defined for {indi_id} {operation=} {value=}")
                transitions[trans] = []
                for action in transition['actions']:
                    if action['type'] not in ('speak', 'file'):
                        log.warning(f"{transition}: {action} type not 'speak' or 'file'?")
                        continue
                    transitions[trans].append(action_dict_to_playback_request(action))
            reactions.append(Reaction(indi_id=indi_id, transitions=transitions))
        return cls(
            reactions=reactions,
            default_voice=default_voice,
            random_utterances=random_utterances,
            soundboard=soundboard,
            walkups=walkups,
        )

if __name__ == "__main__":
    import pprint
    pprint.pprint(Personality.from_path(pathlib.Path('test.toml')), width=255)
