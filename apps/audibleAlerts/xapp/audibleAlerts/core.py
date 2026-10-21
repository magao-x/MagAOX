import datetime
import queue
from typing import Optional, Union
from random import choice
import time
from functools import partial
import logging
import os
import os.path
import pathlib
import re
import xconf
from purepyindi2 import properties, constants, messages
from purepyindi2.messages import DefSwitch, DefText
from magaox.indi.device import XDevice, BaseConfig
from magaox.tts.core import DynamicSpeech, get_or_create_speech_file, duration_from_audio_file

from .personality import Personality, Transition, Recording

log = logging.getLogger(__name__)
HERE = os.path.dirname(__file__)
TAGS_RE = re.compile('<.*?>')

def drop_xml_tags(raw_xml):
  return TAGS_RE.sub('', raw_xml)

@xconf.config
class AudibleAlertsConfig(BaseConfig):
    random_utterance_interval_sec : Union[float, int] = xconf.field(default=15 * 60, help="Seconds since last (real or random) utterance before a random utterance should play")
    cache : pathlib.Path = xconf.field(default=pathlib.Path("/tmp/audibleAlerts_cache"))
    observers_device : str = xconf.field(default='observers', help="Observer controls device name")

def contains_substitutions(text):
    return '{' in text or '}' in text

class AudibleAlerts(XDevice):
    config : AudibleAlertsConfig
    personality : Personality
    _cb_handles : set
    _playback_requests : queue.Queue
    playback_text : properties.TextVector
    personality_sw_prop : properties.SwitchVector = None  # distinguish between initial startup and reload case
    soundboard_sw_prop : properties.SwitchVector = None  # distinguish between initial startup and reload case
    mute : bool = False
    latch_transitions : dict[Transition, constants.AnyIndiValue]  # store last value when triggering a transition so subsequent messages don't trigger too
    per_transition_cooldown_ts : dict[Transition, float]
    last_utterance_ts : float = 0
    last_utterance_chosen : Optional[str] = None
    last_walkup : Optional[dict[str,str]] = None
    last_walkup_ts : float = 0
    walkup_double_trigger_timeout_sec : float = 30

    def enqueue_speech_request(self, sr):
        self._playback_requests.put(sr)

    def handle_speech_text(self, existing_property, new_message):
        if 'target' in new_message and new_message['target'] != existing_property['current']:
            self.log.debug(f"Setting new speech text: {new_message['target']}")
            existing_property['current'] = new_message['target']
            existing_property['target'] = new_message['target']
        self.update_property(existing_property)

    def handle_speak_request(self, existing_property, new_message):
        self.log.debug(f"{new_message['request']=}")
        if new_message['request'] is constants.SwitchState.ON:
            current_text = self.properties['speech_text']['current']
            if current_text is not None and len(current_text.strip()) != 0:
                self.enqueue_speech_request(DynamicSpeech(current_text))
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
            self.log.info("Muted")
        else:
            self.log.info("Unmuted")
        self.update_property(existing_property)
        self.telem("mute_toggle", {"mute": self.mute})

    def walkup_handler(self, new_message):
        which_updated = new_message.name
        for element_name in new_message:
            value = new_message[element_name]
            if value == constants.SwitchState.ON:
                if element_name == self.last_walkup[which_updated]:
                    self.log.debug(f"Already did {self.last_walkup[which_updated]}")
                    return
                if element_name in self.personality.walkups:
                    utterance = choice(self.personality.walkups[element_name])
                    self.log.info(f"Queueing walk-up {utterance} for {element_name}")
                    self.last_walkup_ts = time.time()
                    self.last_walkup[which_updated] = element_name
                    self.enqueue_speech_request(utterance)

    def reaction_handler(self, new_message, element_name, transition, utterance_choices):
        if not isinstance(new_message, messages.IndiSetMessage):
            return
        if element_name not in new_message:
            return
        value = new_message[element_name]
        self.log.debug(f"Judging reaction for {element_name} change to {repr(value)} using {transition}")
        self.log.debug(f"before check {self.latch_transitions=}")
        last_value = self.latch_transitions.get(transition)
        self.log.debug(f"{transition.compare(value)=}, last value was {last_value}, {value != last_value=} {(not transition.compare(last_value))=}")
        if transition.compare(value) and (
            # if there's no operation, we fire on any change,
            # but make sure it's actually a change
            (transition.op is None and value != last_value) or
            # don't fire if we already compared true on the last value:
            (not transition.compare(last_value))
        ):
            self.log.debug(f"Latching {transition}")
            self.latch_transitions[transition] = value
            self.log.debug(f"after update {self.latch_transitions=}")
            self.log.debug(f"latched {transition=} with {value=}")
            # last_transition_ts = self.per_transition_cooldown_ts.get(transition, 0)
            # sec_since_trigger = time.time() - last_transition_ts
            # debounce_expired = sec_since_trigger > transition.debounce_sec
            # self.log.debug(f"Checking for debounce: {sec_since_trigger=} {debounce_expired=}")
            # if debounce_expired:
            utterance = choice(utterance_choices)
            self.log.debug(f"Submitting speech request: {utterance}")
            self.enqueue_speech_request(utterance)
            # else:
                # self.log.debug(f"Would have spoken, but it's only been {sec_since_trigger=}")
        elif transition.compare(last_value) and not transition.compare(value):
            self.log.debug(f"un-latch {transition}, so next time we change to a value that compares True we trigger again. ({last_value=} {value=})")
            self.log.debug(f"before del: {self.latch_transitions=}")
            del self.latch_transitions[transition]
            self.log.debug(f"after del: {self.latch_transitions=}")
        else:
            self.log.debug(f"Got {new_message.device}.{new_message.name} but {transition=} did not match")


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
            self.active_personality = active_personality
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
                self.enqueue_speech_request(srq)
        # set everything off again
        self.update_property(prop)

    def load_personality(self, personality_name):
        personality_file = self.config.common_path_prefix / "config" / "personalities" / f"{personality_name}.toml"
        for cb, device_name, property_name, _ in self._cb_handles:
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

        self._current_voice = self.personality.default_voice

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
                self._cb_handles.add((cb, device_name, property_name, t))
                self.log.debug(f"Registered reaction handler on {device_name=} {property_name=} {element_name=} using transition {t}")
                for idx, utterance in enumerate(reaction.transitions[t]):
                    self.log.debug(f"{reaction.indi_id}: {t}: {utterance}")

        if self.walkup_handler not in self.client.callbacks[self.config.observers_device]['operators']:
            self.client.register_callback(self.walkup_handler, self.config.observers_device, 'operators')
        if self.walkup_handler not in self.client.callbacks[self.config.observers_device]['observers']:
            self.client.register_callback(self.walkup_handler, self.config.observers_device, 'observers')
        self.active_personality = personality_name
        self.telem("load_personality", {'name': personality_name})
        self.send_all_properties()

    def discover_personalities(self):
        personalities_root = self.config.common_path_prefix / "config" / "personalities"
        self.log.debug(f"{personalities_root=}")
        personality_paths = list(personalities_root.glob('*.toml'))
        if len(personality_paths) == 0:
            raise RuntimeError(f"No personality configs found in {personalities_root}, can't do anything")
        personality_shortnames = []
        for fp in personality_paths:
            try:
                shortname = fp.name.rsplit('.', 1)[0]
                if '.' in shortname:
                    raise RuntimeError(f"Base name {shortname} must be valid INDI name (no '.'s)")
                Personality.from_path(fp)
                # if it all worked, this is a valid option
                personality_shortnames.append(shortname)
            except Exception:
                self.log.exception(f"Unable to load {fp}")
                continue
        personality_shortnames.sort()
        return personality_shortnames


    def setup(self):
        self.last_utterance_ts = time.time()
        self.latch_transitions = {}
        self.per_transition_cooldown_ts = {}
        self._cb_handles = set()
        self._playback_requests = queue.Queue()
        self.personality_ids = self.discover_personalities()
        self.active_personality = self.personality_ids[0]
        self.last_walkup = {'observers': '', 'operators': ''}

        while self.client.status is not constants.ConnectionStatus.CONNECTED:
            self.log.info("Waiting for connection...")
            time.sleep(1)
        self.log.info("Connected.")
        self.log.debug(f"Caching synthesis output to {self.config.cache}")
        self.config.cache.mkdir(exist_ok=True)
        

        sv = properties.SwitchVector(
            name="mute",
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        sv.add_element(DefSwitch(name="toggle", _value=constants.SwitchState.ON if self.mute else constants.SwitchState.OFF))
        self.add_property(sv, callback=self.handle_mute_toggle)

        self.personality_sw_prop = properties.SwitchVector(
            name="personality",
            rule=constants.SwitchRule.ONE_OF_MANY,
            perm=constants.PropertyPerm.READ_WRITE,
        )
        for pers in self.personality_ids:
            self.personality_sw_prop.add_element(DefSwitch(name=pers, _value=constants.SwitchState.ON if self.active_personality == pers else constants.SwitchState.OFF))
        self.add_property(self.personality_sw_prop, callback=self.handle_personality_switch)

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
        self.add_property(speech_request, callback=self.handle_speak_request)

        self.playback_text = properties.TextVector(name="playback", perm=constants.PropertyPerm.READ_ONLY)
        self.playback_text.add_element(DefText(
            name="speech",
            _value="",
        ))
        self.playback_text.add_element(DefText(
            name="voice",
            _value="",
        ))
        self.playback_text.add_element(DefText(
            name="file",
            _value="",
        ))
        self.add_property(self.playback_text)

        reload_request = properties.SwitchVector(
            name="reload_personality",
            rule=constants.SwitchRule.ANY_OF_MANY,
        )
        reload_request.add_element(DefSwitch(name="request", _value=constants.SwitchState.OFF))
        self.add_property(reload_request, callback=self.handle_reload_request)

        self.load_personality(self.active_personality)

        self.log.info("Set up complete")

    def loop(self):
        while self._playback_requests.qsize() > 0:
            req = self._playback_requests.get_nowait()
            if self.mute:
                self.log.debug(f"Would have played: {repr(req)}, but muted")
            else:

                if isinstance(req, Recording):
                    self.playback_text['file'] = req.path
                    self.playback_text['speech'] = ''
                    self.playback_text['voice'] = ''
                    self.telem("play", {"file": req.path, "speech": "", "voice": ""})
                    fp = self.config.common_path_prefix / "config" / "personalities" / "data" / req.path
                    if not fp.exists():
                        log.error(f"{fp} doesn't exist, skipping")
                        continue
                elif isinstance(req, DynamicSpeech):
                    # apply substitutions
                    new_req = req.to_speech(self._current_voice, self.client)
                    self.playback_text['file'] = ''
                    self.playback_text['speech'] = new_req.text
                    self.playback_text['voice'] = new_req.voice.name
                    self.telem("play", {"file": "", "speech": new_req.text, "voice": new_req.voice.name})
                    can_cache = new_req.text == req.text
                    fp = get_or_create_speech_file(new_req, can_cache)
                else:
                    raise RuntimeError(f"What is a {repr(req)}?")
                playback_duration_sec = duration_from_audio_file(fp)
                self.log.info(f"Playing: {repr(req)} ({playback_duration_sec:1.1f} sec)")
                self.playback_text.timestamp = datetime.datetime.now()
                self.update_property(self.playback_text)
                self.last_utterance_ts = time.time()  # update timestamp to prevent random utterances
                self.log.debug("Playback request dispatched")

                # give clients time to play it before we do another
                time.sleep(playback_duration_sec)
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
                self.enqueue_speech_request(next_utterance)

# Used to make the pyproject.toml just a little simpler,
# with fewer repetitions of the app name:
main = AudibleAlerts.console_app
