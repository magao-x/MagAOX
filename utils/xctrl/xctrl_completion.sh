# xctrl tab completion for every bash user
#
# Installed by make install in utils/xctrl as /etc/profile.d/xctrl_completion.sh.  The bash-completion package loads
# completions from /usr/share/bash-completion/completions only for users whose ~/.bashrc enables it, so this loads
# xctrl's directly in every bash login shell (and, on Rocky, every interactive one), whatever the user's dotfiles.

if [ -n "${BASH_VERSION:-}" ] && [ -r /usr/share/bash-completion/completions/xctrl ]; then
    . /usr/share/bash-completion/completions/xctrl
fi
