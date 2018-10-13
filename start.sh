export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
workon cv3
cd $HOME/git/SafeT-Rex
git pull
python start.py
