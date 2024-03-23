import sys
from typing import Optional, Union
from random import choice
import time
from functools import partial
from enum import Enum
import logging
import os
import os.path
import pprint
import re
import xconf
from purepyindi2 import device, properties, constants, messages
from purepyindi2.messages import DefNumber, DefSwitch, DefText
from magaox.indi.device import XDevice, BaseConfig

from .personality import Personality, Transition, Operation, SSML, Recording
from .opentts_bridge import speak, ssml_to_wav

log = logging.getLogger(__name__)
HERE = os.path.dirname(__file__)
TAGS_RE = re.compile('<.*?>')

def drop_xml_tags(raw_xml):
  return TAGS_RE.sub('', raw_xml)

@xconf.config
class AudibleAlertsConfig(BaseConfig):
    random_utterance_interval_sec : Union[float, int] = xconf.field(default=15 * 60, help="Seconds since last (real or random) utterance before a random utterance should play")
    cache : xconf.DirectoryConfig = xconf.field(default=xconf.DirectoryConfig(path="/tmp/audibleAlerts_cache"))

def contains_substitutions(text):
    return '{' in text or '}' in text

class AudibleAlerts(XDevice):
    config : AudibleAlertsConfig
    personality : Personality
    _cb_handles : set
    speech_requests : list[str]
    soundboard_sw_prop : properties.SwitchVector = None
    default_voice : str = "coqui-tts:en_ljspeech"  # overridden by personality when loaded
    personalities : list[str] = ['default', 'lab_mode',]
    active_personality : str = "default"
    api_url : str = "http://localhost:5500/"
    mute : bool = False
    latch_transitions : dict[Transition, constants.AnyIndiValue]  # store last value when triggering a transition so subsequent messages don't trigger too
    per_transition_cooldown_ts : dict[Transition, float]
    last_utterance_ts : float = 0
    last_utterance_chosen : Optional[str] = None

    def handle_speech_text(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            self.log.debug(f"Setting new speech text: {new_message['target']}")
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
        self.update_property(existing_property)

    def handle_speech_request(self, existing_property, new_message):
        self.log.debug(f"{new_message['request']=}")
        if new_message['request'] is constants.SwitchState.ON:
            current_text = self.properties['speech_text']['current']
            if current_text is not None and len(current_text.strip()) != 0:
                st = '<speak>' + current_text + '</speak>'
                is_same = [st == getattr(x, 'markup', None) for x in self.speech_requests]
                if any(is_same):
                    self.log.warn(f"Discarding {repr(st)} because it's in the queue already")
                else:
                    self.speech_requests.append(SSML(st))
                    self.log.debug(f"Speech requested: {self.properties['speech_text']['current']}")
                    self.telem("speech_request", {"text": current_text})
        self.update_property(existing_property)  # ensure the request switch turns back off at the client

    def handle_reload_request(self, existing_property, new_message):
        if new_message['request'] is constants.SwitchState.ON:
            self.telem("reload_personality", {"name": self.active_personality})
            self.load_personality(self.active_personality)
        self.update_property(existing_property)  # ensure the request switch turns back off at the client

    def handle_mute_toggle(self, existing_property, new_message):
        existing_property['toggle'] = new_message['toggle']
        self.mute = new_message['toggle'] is constants.SwitchState.ON
        if self.mute:
            log.debug("Muted")
        else:
            log.debug("Unmuted")
        self.update_property(existing_property)
        self.telem("mute_toggle", {"mute": self.mute})

    def reaction_handler(self, new_message, element_name, transition, utterance_choices):
        if not isinstance(new_message, messages.IndiSetMessage):
            return
        if element_name not in new_message:
            return
        value = new_message[element_name]
        self.log.debug(f"Judging reaction for {element_name} change to {repr(value)} using {transition}")
        last_value = self.latch_transitions.get(transition)
        self.log.debug(f"{new_message}\n{transition.compare(value)=}, last value was {last_value}, {value != last_value=} {(not transition.compare(last_value))=}")
        if transition.compare(value) and (
            # if there's no operation, we fire on any change,
            # but make sure it's actually a change
            (transition.op is None and value != last_value) or
            # don't fire if we already compared true on the last value:
            (not transition.compare(last_value))
        ):
            self.latch_transitions[transition] = value
            last_transition_ts = self.per_transition_cooldown_ts.get(transition, 0)
            sec_since_trigger = time.time() - last_transition_ts
            debounce_expired = sec_since_trigger > transition.debounce_sec
            self.log.debug(f"Debounced {sec_since_trigger=}")
            if debounce_expired:
                utterance = choice(utterance_choices)
                self.log.debug(f"Submitting speech request: {utterance}")
                self.speech_requests.append(utterance)
            else:
                self.log.debug(f"Would have talked, but it's only been {sec_since_trigger=}")
        elif transition.compare(last_value) and not transition.compare(value):
            # un-latch, so next time we change to a value that compares True we trigger again:
            del self.latch_transitions[transition]
        else:
            self.log.debug(f"Got {new_message.device}.{new_message.name} but {transition=} did not match")

    def preprocess(self, speech):
        if isinstance(speech, Recording):
            return speech
        speech_text = speech.markup
        substitutables = re.findall(r"({[^}]+})", speech.markup)
        for sub in substitutables:
            indi_id = sub[1:-1]
            value = self.client[indi_id]
            if hasattr(value, 'value'):
                value = value.value
            self.log.debug(f"Replacing {repr(sub)} with {value=}")
            if value is not None:
                try:
                    value = float(value)
                    value = "{:.1f}".format(value)
                except (TypeError, ValueError):
                    value = str(value)
                speech_text = speech_text.replace(sub, value)
        return SSML(speech_text)

    def handle_personality_switch(self, prop : properties.IndiProperty, new_message):
        if not isinstance(new_message, messages.IndiNewMessage):
            return
        active_personality = None
        for elem in prop:
            prop[elem] = constants.SwitchState.OFF
            if elem in new_message and new_message[elem] is constants.SwitchState.ON:
                active_personality = elem

        if active_personality is not None:
            self.log.info(f"Switching to {active_personality=}")
            self.load_personality(active_personality)
        prop[self.active_personality] = constants.SwitchState.ON
        self.update_property(prop)

    def handle_soundboard_switch(self, prop: properties.IndiProperty, new_message):
        print(new_message)
        if not isinstance(new_message, messages.IndiNewMessage):
            return
        for elem in new_message:
            if new_message[elem] is constants.SwitchState.ON:
                srq = self.personality.soundboard[elem]
                self.log.info(f"Soundboard requested {srq}")
                self.speech_requests.append(srq)
        # set everything off again
        self.update_property(prop)

    def load_personality(self, personality_name):
        personality_file = os.path.join(HERE, "personalities", f"{personality_name}.xml")
        for cb, device_name, property_name in self._cb_handles:
            try:
                self.client.unregister_callback(cb, device_name=device_name, property_name=property_name)
            except ValueError:
                log.exception(f"Tried to remove {cb=} {device_name=} {property_name=}")
        self._cb_handles = set()
        if self.soundboard_sw_prop is not None:
            self.delete_property(self.soundboard_sw_prop)

        self.log.info(f"Loading personality from {personality_file}")
        self.personality = Personality.from_path(personality_file)

        self.soundboard_sw_prop = properties.SwitchVector(
            name="soundboard",
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        for btn_name in self.personality.soundboard:
            self.soundboard_sw_prop.add_element(DefSwitch(name=btn_name, _value=constants.SwitchState.OFF))
        self.add_property(self.soundboard_sw_prop, callback=self.handle_soundboard_switch)

        self.default_voice = self.personality.default_voice

        for reaction in self.personality.reactions:
            device_name, property_name, element_name = reaction.indi_id.split('.')
            self.client.get_properties(reaction.indi_id)
            for t in reaction.transitions:
                cb = partial(self.reaction_handler, element_name=element_name, transition=t, utterance_choices=reaction.transitions[t])
                self.client.register_callback(
                    cb,
                    device_name=device_name,
                    property_name=property_name
                )
                self._cb_handles.add((cb, device_name, property_name))
                self.log.debug(f"Registered reaction handler on {device_name=} {property_name=} {element_name=} using transition {t}")
                for idx, utterance in enumerate(reaction.transitions[t]):
                    self.log.debug(f"{reaction.indi_id}: {t}: {utterance}")
                    if isinstance(utterance, SSML):
                        if not contains_substitutions(utterance.markup):
                            result = ssml_to_wav(utterance.markup, self.default_voice, self.api_url, self.config.cache.path)
                            self.log.debug(f"Caching synthesis to {result}")
                        else:
                            self.log.debug(f"Cannot pre-cache because there are substitutions to be made")
        self.active_personality = personality_name
        self.telem("load_personality", {'name': personality_name})

    def setup(self):
        self.last_utterance_ts = time.time()
        self.latch_transitions = {}
        self.per_transition_cooldown_ts = {}
        self._cb_handles = set()
        self.speech_requests = []

        while self.client.status is not constants.ConnectionStatus.CONNECTED:
            self.log.info("Waiting for connection...")
            time.sleep(1)
        self.log.info("Connected.")
        self.log.debug(f"Caching synthesis output to {self.config.cache.path}")
        self.config.cache.ensure_exists()
        self.load_personality(self.active_personality)

        sv = properties.SwitchVector(
            name="mute",
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name=f"toggle", _value=constants.SwitchState.ON if self.mute else constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_mute_toggle)

        sv = properties.SwitchVector(
            name="personality",
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        for pers in self.personalities:
            print(f"{pers=}")
            sv.add_element(DefSwitch(name=pers, _value=constants.SwitchState.ON if self.active_personality == pers else constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_personality_switch)

        speech_text = properties.TextVector(name="speech_text", perm=constants.PropertyPerm.READ_WRITE)
        speech_text.add_element(DefText(
            name="current",
            _value=None,
        ))
        speech_text.add_element(DefText(
            name="target",
            _value=None,
        ))
        self.add_property(speech_text, callback=self.handle_speech_text)

        speech_request = properties.SwitchVector(
            name="speak",
            rule=constants.SwitchRule.ANY_OF_MANY,
        )
        speech_request.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        self.add_property(speech_request, callback=self.handle_speech_request)

        reload_request = properties.SwitchVector(
            name="reload_personality",
            rule=constants.SwitchRule.ANY_OF_MANY,
        )
        reload_request.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        self.add_property(reload_request, callback=self.handle_reload_request)

        self.log.info("Set up complete")

    def loop(self):
        while len(self.speech_requests):
            speech = self.preprocess(self.speech_requests.pop(0))
            if self.mute:
                self.log.debug(f"Would have said: {repr(speech)}, but muted")
            else:
                self.log.info(f"Speaking: {repr(speech)}")
                speak(speech, self.default_voice, self.api_url, self.config.cache.path)
                self.log.debug("Speech complete")
                self.last_utterance_ts = time.time()  # update timestamp to prevent random utterances
        if time.time() - self.last_utterance_ts > self.config.random_utterance_interval_sec and len(self.personality.random_utterances):
            next_utterance = choice(self.personality.random_utterances)
            while next_utterance == self.last_utterance_chosen:
                next_utterance = choice(self.personality.random_utterances)
            self.last_utterance_chosen = next_utterance
            self.last_utterance_ts = time.time()
            if self.mute:
                self.log.debug(f"Would have said: {repr(next_utterance)}, but muted")
            else:
                self.log.info(f"Randomly spouting off: {repr(next_utterance)}")
                speak(next_utterance, self.default_voice, self.api_url, self.config.cache.path)