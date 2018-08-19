#!/bin/bash

flatc -o types/generated/ --cpp types/schemas/git_state.fbs \
                                types/schemas/pdu_outlet_state.fbs \
                                types/schemas/software_log.fbs \
                                types/schemas/state_change.fbs \
                                types/schemas/string_log.fbs 

