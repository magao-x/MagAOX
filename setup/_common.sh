#!/bin/bash
SETUPDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VM_KIND=$(systemd-detect-virt || true)

instrument_user=xsup
instrument_group=magaox
instrument_dev_group=magaox-dev
if [[ $MAGAOX_ROLE == ci || $MAGAOX_ROLE == container ]]; then
  instrument_user=root
  instrument_group=root
  instrument_dev_group=root
fi

function log_error() {
    echo -e "$(tput setaf 1 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}
function exit_with_error() {
  log_error "$1"
  exit 1
}
function log_success() {
    echo -e "$(tput setaf 2 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}
function log_warn() {
    echo -e "$(tput setaf 3 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}
function log_info() {
    echo -e "$(tput setaf 4 2>/dev/null)$1$(tput sgr0 2>/dev/null)"
}

function link_if_necessary() {
  thedir=$1
  thelinkname=$2
  if [[ "$thedir" != "$thelinkname" ]]; then
    if [[ -L $thelinkname ]]; then
      if [[ "$(readlink -- "$thelinkname")" != $thedir ]]; then
        echo "$thelinkname is an existing link, but doesn't point to $thedir. Aborting."
        return 1
      fi
    elif [[ -e $thelinkname ]]; then
      echo "$thelinkname exists, but is not a symlink and we want the destination to be $thedir."
      return 1
    else
      sudo ln -sv "$thedir" "$thelinkname"
    fi
  fi
}

function setgid_all() {
    # n.b. can't be recursive because g+s on files means something else
    # so we find all directories and individually chmod them:
    if [[ "$EUID" != 0 ]]; then
      find $1 -type d -exec sudo chmod g+rxs {} \; || return 1
    else
      find $1 -type d -exec chmod g+rxs {} \; || return 1
    fi
}

function make_on_data_array() {
  # If run on instrument computer, make the name provided as an arg a link from $2/$1
  # to /data/$1.
  # If not on a real instrument computer, just make a normal folder under /opt/MagAOX/
  if [[ -z $1 ]]; then
    log_error "Missing target name argument for make_on_data_array"
    return 1
  else
    TARGET_NAME=$1
  fi
  if [[ -z $2 ]]; then
    log_error "Missing parent dir argument for make_on_data_array"
    return 1
  else
    PARENT_DIR=$2
  fi

  if [[ $MAGAOX_ROLE == RTC || $MAGAOX_ROLE == ICC || $MAGAOX_ROLE == AOC ]]; then
    REAL_DIR=/data/$TARGET_NAME
    sudo mkdir -pv $REAL_DIR || return 1
    link_if_necessary $REAL_DIR $PARENT_DIR/$TARGET_NAME || return 1
  else
    REAL_DIR=$PARENT_DIR/$TARGET_NAME
    sudo mkdir -pv $REAL_DIR || return 1
  fi

  sudo chown -RP $instrument_user:$instrument_group $REAL_DIR || return 1
  sudo chmod -R u=rwX,g=rwX,o=rX $REAL_DIR || return 1
  setgid_all $REAL_DIR || return 1
}

function _cached_fetch() {
  url=$1
  filename=$2
  dest=$PWD
  if [[ $filename == *levmar* ]]; then
    EXTRA_CURL_OPTS='-A "Mozilla/5.0"'
  else
    EXTRA_CURL_OPTS=''
  fi
  mkdir -p /opt/MagAOX/.cache || return 1
  if [[ ! -e $dest/$filename ]]; then
    if [[ ! -e /opt/MagAOX/.cache/$filename ]]; then
      curl $EXTRA_CURL_OPTS --fail -L $url > /tmp/magaoxcache-$filename || return 1
      mv /tmp/magaoxcache-$filename /opt/MagAOX/.cache/$filename || return 1
      log_info "Downloaded to /opt/MagAOX/.cache/$filename"
    fi
    cp /opt/MagAOX/.cache/$filename $dest/$filename || return 1
    log_info "Copied to $dest/$filename"
  fi
}

function normalize_git_checkout() {
  destdir=$1
  pushd $destdir || return 1
  git config core.sharedRepository group || return 1
  git submodule update --init --recursive
  sudo chown -R :$instrument_dev_group $destdir || return 1
  sudo chmod -R g=rwX $destdir || return 1
  # n.b. can't be recursive because g+s on files means something else
  # so we find all directories and individually chmod them:
  sudo find $destdir -type d -exec chmod g+s {} \; || return 1
  log_success "Normalized permissions on $destdir"
  popd || return 1
}

function clone_or_update_and_cd() {
    orgname=$1
    reponame=$2
    parentdir=$3
    # Disable unbound var check because we do it ourselves:
    if [[ "$SHELLOPTS" =~ "nounset" ]]; then _WAS_NOUNSET=1; set +u; fi
    if [[ ! -z $4 ]]; then
      destdir=$4
    else
      destdir="$parentdir/$reponame"
    fi
    if [[ $_WAS_NOUNSET == 1 ]]; then set -u; fi
    # and re-enable.

    if [[ ! -d $parentdir/$reponame/.git ]]; then
      echo "Cloning new copy of $orgname/$reponame"
      CLONE_DEST=/tmp/${reponame}_$(date +"%s")
      git clone https://github.com/$orgname/$reponame.git $CLONE_DEST || return 1
      sudo rsync -a $CLONE_DEST/ $destdir/ || return 1
      cd $destdir/ || return 1
      log_success "Cloned new $destdir"
      rm -rf $CLONE_DEST
      log_success "Removed temporary clone at $CLONE_DEST"
    else
      cd $destdir
      if [[ "$(git rev-parse --abbrev-ref --symbolic-full-name HEAD)" != HEAD ]]; then
        git pull --ff-only || return 1
      else
        git fetch || return 1
        log_info "Not pulling because a specific commit is checked out"
      fi
      log_success "Updated $destdir"
    fi
    normalize_git_checkout $destdir || return 1
}

DEFAULT_PASSWORD="extremeAO!"

function creategroup() {
  if [[ ! $(getent group $1) ]]; then
    sudo groupadd $1 || return 1
    echo "Added group $1"
  else
    echo "Group $1 exists"
  fi
}

function createuser() {
  username=$1
  if getent passwd $username > /dev/null 2>&1; then
    log_info "User account $username exists"
  else
    sudo useradd -U $username || return 1
    echo -e "$DEFAULT_PASSWORD\n$DEFAULT_PASSWORD" | sudo -H passwd $username || exit 1
    log_success "Created user account $username with default password $DEFAULT_PASSWORD"
  fi
  sudo usermod -a -G magaox $username || return 1
  log_info "Added user $username to group magaox"
  sudo mkdir -p /home/$username/.ssh || return 1
  sudo touch /home/$username/.ssh/authorized_keys || return 1
  sudo chmod -R u=rwx,g=,o= /home/$username/.ssh || return 1
  sudo chmod u=rw,g=,o= /home/$username/.ssh/authorized_keys || return 1
  sudo chown -R $username:$instrument_group /home/$username || return 1
  sudo -H chsh $username -s $(which bash) || return 1
  log_info "Append an ecdsa or ed25519 key to /home/$username/.ssh/authorized_keys to enable SSH login"

  data_path="/data/users/$username/"
  sudo -H mkdir -p "$data_path" || return 1
  sudo -H chown "$username:$instrument_group" "$data_path" || return 1
  sudo -H chmod g+rxs "$data_path" || return 1
  log_success "Created $data_path"

  link_name="/home/$username/data"
  if sudo test ! -L "$link_name"; then
    sudo -H ln -sv "$data_path" "$link_name" || return 1
    log_success "Linked $link_name -> $data_path"
  fi
}
# We work around the buggy devtoolset /bin/sudo wrapper in provision.sh, but
# that means we have to explicitly enable it ourselves.
# (This crap again: https://bugzilla.redhat.com/show_bug.cgi?id=1319936)
# Also, we control whether to build with the devtoolset gcc7 compiler or the
# included gcc4 compiler with $BUILDING_KERNEL_STUFF. If that's set to 1,
# use the included gcc4 for consistency with the running kernel.
if [[ "$SHELLOPTS" =~ "nounset" ]]; then _WAS_NOUNSET=1; set +u; fi # Temporarily disable checking for unbound variables to set a default value
  if [[ -z $BUILDING_KERNEL_STUFF ]]; then BUILDING_KERNEL_STUFF=0; fi
if [[ $_WAS_NOUNSET == 1 ]]; then set -u; fi

if [[ $BUILDING_KERNEL_STUFF != 1 && -e /opt/rh/devtoolset-7/enable ]]; then
  if [[ "$SHELLOPTS" =~ "nounset" ]]; then _WAS_NOUNSET=1; set +u; fi
    source /opt/rh/devtoolset-7/enable
  if [[ $_WAS_NOUNSET == 1 ]]; then set -u; fi
fi
# root doesn't get /usr/local/bin on their path, so add it
# (why? https://serverfault.com/a/838552)
if [[ $PATH != "*/usr/local/bin*" ]]; then
    export PATH="/usr/local/bin:$PATH"
fi
if [[ $(which sudo) == *devtoolset* ]]; then
  REAL_SUDO=/usr/bin/sudo
else
  REAL_SUDO=$(which sudo)
fi
